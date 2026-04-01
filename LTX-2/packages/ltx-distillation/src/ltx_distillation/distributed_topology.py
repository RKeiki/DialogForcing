from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import torch.distributed as dist


def _new_group(
    ranks: List[int],
    *,
    backend: Optional[str] = None,
) -> Optional[dist.ProcessGroup]:
    if not dist.is_initialized() or len(ranks) <= 1:
        return None
    kwargs = {"ranks": ranks}
    if backend is not None:
        kwargs["backend"] = backend
    return dist.new_group(**kwargs)


@dataclass
class DistributedTopology:
    enabled: bool
    role: str
    global_rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    node_rank: int
    num_nodes: int
    teacher_ranks: List[int]
    worker_groups: List[List[int]]
    model_ranks: List[int]
    model_process_group: Optional[dist.ProcessGroup]
    worker_replica_process_group: Optional[dist.ProcessGroup]
    control_process_group: Optional[dist.ProcessGroup]
    worker_group_index: Optional[int]
    worker_group_rank: Optional[int]
    num_worker_groups: int
    worker_global_rank: Optional[int]
    worker_global_world_size: int
    teacher_peer_rank: Optional[int]
    teacher_service_source_ranks: List[int]
    is_primary_worker_group: bool
    is_primary_worker_leader: bool

    @property
    def is_teacher(self) -> bool:
        return self.role == "teacher"

    @property
    def is_worker(self) -> bool:
        return self.role == "worker"

    @property
    def worker_group_world_size(self) -> int:
        return len(self.model_ranks)

    @property
    def worker_slot(self) -> Optional[int]:
        if self.worker_group_rank is None:
            return None
        return self.worker_group_rank


def build_distributed_topology(
    *,
    config,
    rank: int,
    world_size: int,
    local_rank: int,
) -> DistributedTopology:
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", max(local_rank + 1, 1)))
    if world_size % local_world_size != 0:
        raise ValueError(
            f"world_size={world_size} is not divisible by LOCAL_WORLD_SIZE={local_world_size}"
        )

    num_nodes = world_size // local_world_size
    node_rank = rank // local_world_size

    split_enabled = bool(getattr(config, "split_teacher_worker", False))
    if not split_enabled:
        return DistributedTopology(
            enabled=False,
            role="worker",
            global_rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
            node_rank=node_rank,
            num_nodes=num_nodes,
            teacher_ranks=[],
            worker_groups=[list(range(world_size))],
            model_ranks=list(range(world_size)),
            model_process_group=None,
            worker_replica_process_group=None,
            control_process_group=_new_group(list(range(world_size)), backend="gloo"),
            worker_group_index=0,
            worker_group_rank=rank,
            num_worker_groups=1,
            worker_global_rank=rank,
            worker_global_world_size=world_size,
            teacher_peer_rank=None,
            teacher_service_source_ranks=[],
            is_primary_worker_group=True,
            is_primary_worker_leader=rank == 0,
        )

    teacher_num_nodes = int(getattr(config, "teacher_num_nodes", 1))
    worker_num_nodes_per_group = int(getattr(config, "worker_num_nodes_per_group", 1))
    if teacher_num_nodes <= 0:
        raise ValueError("split_teacher_worker=true requires teacher_num_nodes >= 1")
    if worker_num_nodes_per_group <= 0:
        raise ValueError("worker_num_nodes_per_group must be >= 1")
    if teacher_num_nodes >= num_nodes:
        raise ValueError(
            f"teacher_num_nodes={teacher_num_nodes} leaves no worker nodes out of num_nodes={num_nodes}"
        )

    teacher_rank_count = teacher_num_nodes * local_world_size
    remaining_nodes = num_nodes - teacher_num_nodes
    if remaining_nodes % worker_num_nodes_per_group != 0:
        raise ValueError(
            f"Remaining nodes={remaining_nodes} is not divisible by "
            f"worker_num_nodes_per_group={worker_num_nodes_per_group}"
        )

    teacher_ranks = list(range(teacher_rank_count))
    worker_groups: List[List[int]] = []
    worker_group_rank_count = worker_num_nodes_per_group * local_world_size
    worker_start = teacher_rank_count
    while worker_start < world_size:
        worker_groups.append(list(range(worker_start, worker_start + worker_group_rank_count)))
        worker_start += worker_group_rank_count

    if not worker_groups:
        raise ValueError("split_teacher_worker=true produced no worker groups")

    if len(teacher_ranks) != worker_group_rank_count:
        raise ValueError(
            "Remote teacher service currently requires teacher group size to match each worker "
            f"group size, got teacher_ranks={len(teacher_ranks)} and worker_group_size={worker_group_rank_count}"
        )

    teacher_pg = _new_group(teacher_ranks)
    worker_pgs = [_new_group(ranks) for ranks in worker_groups]
    control_pg = _new_group(list(range(world_size)), backend="gloo")
    replica_pgs = []
    for slot in range(worker_group_rank_count):
        replica_ranks = [group[slot] for group in worker_groups]
        replica_pgs.append(_new_group(replica_ranks))

    role = "teacher" if rank in teacher_ranks else "worker"
    worker_group_index: Optional[int] = None
    worker_group_rank: Optional[int] = None
    model_ranks: List[int]
    model_process_group: Optional[dist.ProcessGroup]
    worker_replica_process_group: Optional[dist.ProcessGroup] = None
    worker_global_rank: Optional[int] = None
    teacher_peer_rank: Optional[int] = None
    teacher_service_source_ranks: List[int] = []

    flattened_worker_ranks = [worker_rank for group in worker_groups for worker_rank in group]

    if role == "teacher":
        model_ranks = teacher_ranks
        model_process_group = teacher_pg
        teacher_slot = teacher_ranks.index(rank)
        teacher_service_source_ranks = [group[teacher_slot] for group in worker_groups]
    else:
        for idx, group in enumerate(worker_groups):
            if rank in group:
                worker_group_index = idx
                worker_group_rank = group.index(rank)
                model_ranks = group
                model_process_group = worker_pgs[idx]
                worker_replica_process_group = replica_pgs[worker_group_rank]
                worker_global_rank = flattened_worker_ranks.index(rank)
                teacher_peer_rank = teacher_ranks[worker_group_rank]
                break
        else:
            raise RuntimeError(f"Rank {rank} is in neither teacher nor worker groups")

    return DistributedTopology(
        enabled=True,
        role=role,
        global_rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
        node_rank=node_rank,
        num_nodes=num_nodes,
        teacher_ranks=teacher_ranks,
        worker_groups=worker_groups,
        model_ranks=model_ranks,
        model_process_group=model_process_group,
        worker_replica_process_group=worker_replica_process_group,
        control_process_group=control_pg,
        worker_group_index=worker_group_index,
        worker_group_rank=worker_group_rank,
        num_worker_groups=len(worker_groups),
        worker_global_rank=worker_global_rank,
        worker_global_world_size=len(flattened_worker_ranks),
        teacher_peer_rank=teacher_peer_rank,
        teacher_service_source_ranks=teacher_service_source_ranks,
        is_primary_worker_group=(role == "worker" and worker_group_index == 0),
        is_primary_worker_leader=(role == "worker" and worker_group_index == 0 and worker_group_rank == 0),
    )
