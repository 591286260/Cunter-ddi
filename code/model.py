"""CPST-DDI model and differentiable partial structure transport."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def masked_mean(x, mask, dim=1):
    weight = mask.to(x.dtype).unsqueeze(-1)
    return (x * weight).sum(dim) / weight.sum(dim).clamp_min(1.0)


class TokenEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class PartialFusedTransport(nn.Module):
    """Entropic partial FGW solver unrolled for end-to-end optimization.

    Partial transport is approximated with learned source/target participation
    gates followed by masked Sinkhorn normalization. The FGW structural cost is
    updated from the previous plan; gradients flow through all updates.
    """
    def __init__(self, hidden_dim, temperature=0.15, mass=0.7, gw_weight=0.3, sinkhorn_iters=8):
        super().__init__()
        self.temperature = temperature
        self.mass = mass
        self.gw_weight = gw_weight
        self.sinkhorn_iters = sinkhorn_iters
        self.source_gate = nn.Linear(hidden_dim, 1)
        self.target_gate = nn.Linear(hidden_dim, 1)
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _sinkhorn(self, cost, source_mass, target_mass, valid):
        log_k = -cost / self.temperature
        log_k = log_k.masked_fill(~valid, -1e4)
        log_u = torch.zeros_like(source_mass)
        log_v = torch.zeros_like(target_mass)
        log_a = source_mass.clamp_min(1e-8).log()
        log_b = target_mass.clamp_min(1e-8).log()
        for _ in range(self.sinkhorn_iters):
            log_u = log_a - torch.logsumexp(log_k + log_v[:, None, :], dim=2)
            log_v = log_b - torch.logsumexp(log_k + log_u[:, :, None], dim=1)
        plan = torch.exp(log_k + log_u[:, :, None] + log_v[:, None, :])
        return plan * valid.to(plan.dtype)

    def forward(self, source, target, source_mask, target_mask, source_dist, target_dist):
        source_norm = F.normalize(self.q(source), dim=-1)
        target_norm = F.normalize(self.k(target), dim=-1)
        semantic_cost = 1.0 - torch.bmm(source_norm, target_norm.transpose(1, 2))
        valid = source_mask[:, :, None] & target_mask[:, None, :]

        # Participation gates allow irrelevant tokens to retain little mass.
        a = torch.sigmoid(self.source_gate(source).squeeze(-1)) * source_mask
        b = torch.sigmoid(self.target_gate(target).squeeze(-1)) * target_mask
        a = a / a.sum(1, keepdim=True).clamp_min(1e-8) * self.mass
        b = b / b.sum(1, keepdim=True).clamp_min(1e-8) * self.mass
        plan = self._sinkhorn(semantic_cost, a, b, valid)

        if self.gw_weight > 0:
            for _ in range(2):
                # Squared-loss GW tensor: A^2 a + B^2 b - 2 A T B^T.
                row_mass = plan.sum(2)
                col_mass = plan.sum(1)
                left = torch.bmm(source_dist.square(), row_mass.unsqueeze(-1))
                right = torch.bmm(target_dist.square(), col_mass.unsqueeze(-1)).transpose(1, 2)
                cross = torch.bmm(torch.bmm(source_dist, plan), target_dist.transpose(1, 2))
                structural_cost = (left + right - 2.0 * cross).clamp_min(0.0)
                cost = (1.0 - self.gw_weight) * semantic_cost + self.gw_weight * structural_cost
                plan = self._sinkhorn(cost, a, b, valid)
        return plan, semantic_cost


class CPSTDDI(nn.Module):
    def __init__(self, payload, cfg):
        super().__init__()
        h, drop = cfg["hidden_dim"], cfg["dropout"]
        self.cfg = cfg
        self.max_motifs = cfg["max_motifs"]
        self.mol_encoder = TokenEncoder(32, h, drop)
        self.drug_embedding = nn.Embedding(len(payload["records"]), h)
        self.fp_encoder = nn.Sequential(nn.Linear(1024, h * 2), nn.LayerNorm(h * 2), nn.GELU(), nn.Dropout(drop), nn.Linear(h * 2, h))
        self.legacy_pair_without_struct = cfg.get("legacy_pair_without_struct", False)
        if not self.legacy_pair_without_struct:
            self.struct_encoder = nn.Sequential(nn.Linear(8, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, h))
        self.drug_fuse = nn.Sequential(nn.Linear(h * 3, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(drop))
        self.rel_embedding = nn.Embedding(payload["num_kg_relations"] + 1, h)
        self.dir_embedding = nn.Embedding(2, h)
        # Hashing bounds memory while retaining neighbor identity information.
        self.neighbor_buckets = min(65536, max(4096, payload["num_entities"]))
        self.neighbor_embedding = nn.Embedding(self.neighbor_buckets, h)
        self.kg_norm = nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(drop))
        self.ddi_relation = nn.Embedding(payload["num_ddi_relations"] + 1, h)
        self.relation_bilinear = nn.Linear(h, h, bias=False)
        self.mol_condition = nn.MultiheadAttention(h, num_heads=4, dropout=drop, batch_first=True)
        self.transport = PartialFusedTransport(
            h, cfg["transport_temperature"], cfg["transport_mass"], cfg["gw_weight"], cfg["sinkhorn_iters"]
        )
        self.pair_dim = h * (3 if self.legacy_pair_without_struct else 4)
        self.match_encoder = nn.Sequential(nn.Linear(h * 5, h * 2), nn.GELU(), nn.Dropout(drop), nn.Linear(h * 2, h))
        decoder_in = self.pair_dim + h + h
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, h * 2), nn.LayerNorm(h * 2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(h * 2, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, 1),
        )

    def _encode(self, batch):
        mol = self.mol_encoder(batch["mol"])
        split = self.max_motifs
        a, b = mol[:, :split], mol[:, split:]
        am, bm = batch["mol_mask"][:, :split], batch["mol_mask"][:, split:]
        # Pair-conditioned refinement; padding keys are masked.
        a_ctx, _ = self.mol_condition(a, b, b, key_padding_mask=~bm)
        b_ctx, _ = self.mol_condition(b, a, a, key_padding_mask=~am)
        mol = torch.cat([a + a_ctx, b + b_ctx], dim=1)
        ga, gb = masked_mean(mol[:, :split], am), masked_mean(mol[:, split:], bm)
        ida = self.drug_embedding(batch["drug_index"][:, 0])
        idb = self.drug_embedding(batch["drug_index"][:, 1])
        if not self.cfg.get("use_drug_id", True):
            ida = torch.zeros_like(ida)
            idb = torch.zeros_like(idb)
        fpa = self.fp_encoder(batch["fingerprint"][:, 0])
        fpb = self.fp_encoder(batch["fingerprint"][:, 1])
        da = self.drug_fuse(torch.cat([ga, fpa, ida], dim=-1))
        db = self.drug_fuse(torch.cat([gb, fpb, idb], dim=-1))
        pair_parts = [da + db, (da - db).abs(), da * db]
        if not self.legacy_pair_without_struct:
            pair_struct = self.struct_encoder(batch["pair_struct"])
            if not self.cfg.get("use_topology", True):
                pair_struct = torch.zeros_like(pair_struct)
            pair_parts.append(pair_struct)
        pair_global = torch.cat(pair_parts, dim=-1)

        kg = (
            self.rel_embedding(batch["kg_rel"])
            + self.dir_embedding(batch["kg_dir"])
        )
        if self.cfg.get("use_neighbor_id", True):
            kg = kg + self.neighbor_embedding(batch["kg_nei"] % self.neighbor_buckets)
        kg = self.kg_norm(kg)
        kg_global = masked_mean(kg, batch["kg_mask"])
        relation = self.ddi_relation(batch["relation"])
        bilinear = (da * self.relation_bilinear(relation) * db).sum(-1) / (da.shape[-1] ** 0.5)
        return mol, kg, pair_global, kg_global, relation, bilinear

    def _decode_plan(self, mol, kg, plan, pair_global, kg_global, relation):
        # Avoid explicitly materializing every 5h pair feature in one huge tensor.
        transported_kg = torch.bmm(plan, kg)
        row_mass = plan.sum(2, keepdim=True).clamp_min(1e-8)
        transported_kg = transported_kg / row_mass
        rel = relation[:, None, :].expand_as(mol)
        match = self.match_encoder(torch.cat([mol, transported_kg, mol * transported_kg, (mol - transported_kg).abs(), rel], -1))
        weight = plan.sum(2)
        mechanism = (match * weight.unsqueeze(-1)).sum(1) / weight.sum(1, keepdim=True).clamp_min(1e-8)
        return self.decoder(torch.cat([pair_global, mechanism, kg_global], -1)).squeeze(-1)

    def forward(self, batch, counterfactual=False):
        mol, kg, pair_global, kg_global, relation, bilinear = self._encode(batch)
        if not self.cfg["use_kg"]:
            kg_global = torch.zeros_like(kg_global)
        if not self.cfg["use_transport"]:
            mechanism = torch.zeros_like(kg_global)
            logits = self.decoder(torch.cat([pair_global, mechanism, kg_global], -1)).squeeze(-1)
            return {"logits": logits + bilinear}

        plan, cost = self.transport(
            mol, kg, batch["mol_mask"], batch["kg_mask"], batch["mol_dist"], batch["kg_dist"]
        )
        logits = self._decode_plan(mol, kg, plan, pair_global, kg_global, relation) + bilinear
        output = {"logits": logits, "plan": plan, "cost": cost}
        if counterfactual and self.cfg["use_counterfactual"]:
            valid = batch["mol_mask"][:, :, None] & batch["kg_mask"][:, None, :]
            # A fully vectorized threshold avoids one GPU synchronization per
            # sample. Samples with fewer valid cells simply keep all of them.
            flat = plan.flatten(1)
            count = max(1, int(flat.shape[1] * self.cfg["keep_ratio"]))
            threshold = flat.topk(count, dim=1).values[:, -1]
            core = (plan >= threshold[:, None, None]) & valid
            keep = plan * core
            remove = plan * (~core)
            output["keep_logits"] = self._decode_plan(mol, kg, keep, pair_global, kg_global, relation) + bilinear
            output["remove_logits"] = self._decode_plan(mol, kg, remove, pair_global, kg_global, relation) + bilinear
        return output
