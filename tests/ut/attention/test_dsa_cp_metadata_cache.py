# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import torch

import vllm_ascend.attention.context_parallel.dsa_cp as dsa_cp
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.dsa_cp import (
    AscendDSACPMetadataBuilder,
    _build_a5_c1_sas_signature,
)
from vllm_ascend.device.device_op import DeviceOperator


def test_a5_c1_sas_signature_uses_semantic_window_buckets():
    q1 = torch.tensor([0, 1], dtype=torch.int32)
    q4 = torch.tensor([0, 4], dtype=torch.int32)

    assert _build_a5_c1_sas_signature(q1, torch.tensor([64]), 1) == ((1, 1),)
    assert _build_a5_c1_sas_signature(q1, torch.tensor([65]), 1) == ((1, 0),)
    assert _build_a5_c1_sas_signature(q4, torch.tensor([64]), 1) == ((4, 4),)
    assert _build_a5_c1_sas_signature(q4, torch.tensor([65]), 1) == ((4, 3),)
    assert _build_a5_c1_sas_signature(q4, torch.tensor([68]), 1) == ((4, 0),)
    assert _build_a5_c1_sas_signature(q1, torch.tensor([65]), 1) == _build_a5_c1_sas_signature(
        q1, torch.tensor([4096]), 1
    )


def test_a5_c1_sas_signature_preserves_mtp_request_order_and_empty_rows():
    q_4_1 = torch.tensor([0, 4, 5], dtype=torch.int32)
    q_1_4 = torch.tensor([0, 1, 5], dtype=torch.int32)
    q_1_0_1 = torch.tensor([0, 1, 1, 2], dtype=torch.int32)
    q_1_1_0 = torch.tensor([0, 1, 2, 2], dtype=torch.int32)

    assert _build_a5_c1_sas_signature(q_4_1, torch.tensor([68, 65]), 2) != _build_a5_c1_sas_signature(
        q_1_4, torch.tensor([65, 68]), 2
    )
    assert _build_a5_c1_sas_signature(q_1_0_1, torch.tensor([65, 123, 65]), 3) != _build_a5_c1_sas_signature(
        q_1_1_0, torch.tensor([65, 65, 123]), 3
    )
    assert _build_a5_c1_sas_signature(q_1_0_1, torch.tensor([65, 123, 65]), 3) == _build_a5_c1_sas_signature(
        q_1_0_1, torch.tensor([65, 999, 65]), 3
    )


def test_a5_c1_sas_signature_falls_back_outside_proven_domain():
    assert _build_a5_c1_sas_signature(torch.tensor([0, 17]), torch.tensor([17]), 1) is None
    assert _build_a5_c1_sas_signature(torch.tensor([1, 0]), torch.tensor([1]), 1) is None
    assert _build_a5_c1_sas_signature(torch.tensor([0, 4]), torch.tensor([3]), 1) is None
    assert _build_a5_c1_sas_signature(torch.tensor([0]), torch.tensor([], dtype=torch.int32), 1) is None


def _make_c1_builder() -> AscendDSACPMetadataBuilder:
    builder = object.__new__(AscendDSACPMetadataBuilder)
    builder.compressor_ratio = 0
    builder.common_ratio_to_sas_metadata = {}
    builder._a5_c1_sas_cache_enabled = True
    builder._c1_sas_cache_signature = None
    builder._c1_sas_cache_metadata = None
    builder._c1_sas_cache_hits = 0
    builder._c1_sas_cache_misses = 0
    builder.req_sas_metadata = torch.zeros(1024, dtype=torch.int32)
    builder.seqused_q = torch.tensor([], dtype=torch.int32)
    builder._zero_i32 = torch.tensor([0], dtype=torch.int32)
    builder.cu_seqlens_ori_kv = torch.tensor([], dtype=torch.int32)
    builder.cu_seqlens_cmp_kv = torch.tensor([], dtype=torch.int32)
    builder.model_config = mock.MagicMock()
    builder.model_config.get_head_size.return_value = 128
    builder.model_config.hf_config.sliding_window = 128
    return builder


def _build_c1_sas(builder: AscendDSACPMetadataBuilder, seq_len: int) -> torch.Tensor:
    builder.common_ratio_to_sas_metadata = {}
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    return builder._build_sas_metadata(
        num_heads=128,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        query_start_loc_cpu=query_start_loc,
        seq_lens_cpu=seq_lens,
        max_query_len=1,
        max_seq_lens=seq_len,
        index_topk=512,
        num_reqs=1,
        has_prefill=False,
        cu_cmp_seqlen_list=None,
    )


def test_a5_c1_sas_cache_hits_across_rounds_and_keeps_stable_buffer():
    builder = _make_c1_builder()
    generated = [
        torch.full((1024,), 11, dtype=torch.int32),
        torch.full((1024,), 22, dtype=torch.int32),
    ]
    metadata_op = mock.MagicMock(side_effect=generated)

    with (
        mock.patch.object(DeviceOperator, "get_dsa_sparse_attn_metadata_op", return_value=metadata_op),
        mock.patch.object(DeviceOperator, "get_dsa_sparse_attn_metadata_kwargs", return_value={}),
        mock.patch.object(DeviceOperator, "get_dsa_decode_cu_seqlens_ori_kv", return_value=None),
        mock.patch.object(DeviceOperator, "get_dsa_decode_cu_seqlens_cmp_kv", return_value=None),
    ):
        stable_data_ptr = builder.req_sas_metadata.data_ptr()
        first = _build_c1_sas(builder, seq_len=65)
        first_value = first.clone()
        builder.req_sas_metadata.fill_(-1)
        hit = _build_c1_sas(builder, seq_len=4096)
        hit_value = hit.clone()
        miss = _build_c1_sas(builder, seq_len=64)
        miss_value = miss.clone()

    assert metadata_op.call_count == 2
    assert builder.get_c1_sas_cache_stats() == (1, 2)
    assert first.data_ptr() == stable_data_ptr
    assert hit.data_ptr() == stable_data_ptr
    assert miss.data_ptr() == stable_data_ptr
    assert torch.equal(first_value, generated[0])
    assert torch.equal(hit_value, generated[0])
    assert torch.equal(miss_value, generated[1])


def test_qli_metadata_uses_precomputed_cpu_maxima():
    builder = object.__new__(AscendDSACPMetadataBuilder)
    builder.compressor_ratio = 4
    builder.common_ratio_to_sas_metadata = {}
    builder.req_qli_metadata = torch.zeros(1024, dtype=torch.int32)
    builder.seqused_q = torch.tensor([], dtype=torch.int32)
    builder.model_config = mock.MagicMock()
    builder.model_config.hf_config.index_n_heads = 64
    builder.model_config.hf_config.index_head_dim = 128
    builder.model_config.hf_config.index_topk = 512

    fake_ascend_ops = mock.MagicMock()
    fake_ascend_ops.npu_vllm_quant_lightning_indexer_metadata.return_value = torch.ones(1024, dtype=torch.int32)
    with mock.patch.object(torch.ops, "_C_ascend", fake_ascend_ops):
        builder._build_qli_metadata(
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([4096], dtype=torch.int32),
            max_seqlen_q=7,
            max_seqlen_k=123,
            num_reqs=1,
        )

    call_kwargs = fake_ascend_ops.npu_vllm_quant_lightning_indexer_metadata.call_args.kwargs
    assert call_kwargs["max_seqlen_q"] == 7
    assert call_kwargs["max_seqlen_k"] == 123


def test_build_req_metadata_passes_cpu_maxima_to_qli():
    builder = _make_c1_builder()
    builder.seq_lens = torch.tensor([123], dtype=torch.int32)
    builder.seq_lens_cpu = torch.tensor([123], dtype=torch.int32)
    builder.start_pos_prefill = torch.zeros(1, dtype=torch.int32)
    builder.block_table = torch.zeros((1, 1), dtype=torch.int32)
    builder.slot_mapping = torch.zeros(4, dtype=torch.int32)
    builder.num_actual_tokens = 4
    builder.local_query_start_loc = torch.zeros(2, dtype=torch.int32)
    builder.local_seq_lens = torch.zeros(1, dtype=torch.int32)
    builder.model_config.hf_config.num_attention_heads = 128
    builder.model_config.hf_config.index_topk = 512

    query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
    local_seq_lens = torch.tensor([123], dtype=torch.int32)
    local_metadata = (
        0,
        4,
        4,
        4,
        query_start_loc,
        local_seq_lens,
        torch.zeros((4, 1)),
        torch.zeros((4, 1)),
    )
    builder._build_local_token_metadata = mock.MagicMock(side_effect=[local_metadata, local_metadata])
    builder._get_cmp_seqlens_for_metadata = mock.MagicMock(return_value=None)
    builder._build_sas_metadata = mock.MagicMock(return_value=torch.zeros(1024, dtype=torch.int32))
    builder._build_qli_metadata = mock.MagicMock(return_value=None)

    common_attn_metadata = mock.MagicMock()
    common_attn_metadata.num_reqs = 1
    common_attn_metadata.query_start_loc = query_start_loc
    common_attn_metadata.query_start_loc_cpu = query_start_loc
    builder.build_req_metadata(
        common_attn_metadata=common_attn_metadata,
        input_positions=torch.arange(4),
        cos=torch.zeros((4, 1)),
        sin=torch.zeros((4, 1)),
        num_input_tokens=4,
        num_reqs_actual=1,
        attn_state=AscendAttentionState.DecodeOnly,
    )

    builder._build_qli_metadata.assert_called_once()
    call_kwargs = builder._build_qli_metadata.call_args.kwargs
    assert call_kwargs["query_start_loc"] is query_start_loc
    assert call_kwargs["seq_lens"] is local_seq_lens
    assert call_kwargs["max_seqlen_q"] == 4
    assert call_kwargs["max_seqlen_k"] == 123
    assert call_kwargs["num_reqs"] == 1


def test_draft_local_rope_uses_isolated_cached_buffer_per_draft_index():
    builder = object.__new__(AscendDSACPMetadataBuilder)
    query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    seq_lens = torch.tensor([3], dtype=torch.int32)
    input_positions = torch.arange(3, dtype=torch.int64)
    spec_buffers = [
        (
            torch.full((3, 1, 1, 2), 11.0),
            torch.full((3, 1, 1, 2), -11.0),
        ),
        (
            torch.full((3, 1, 1, 2), 22.0),
            torch.full((3, 1, 1, 2), -22.0),
        ),
    ]
    rope_calls = []
    call_values = ((11.0, -11.0), (22.0, -22.0), (33.0, -33.0))

    def fake_get_cos_and_sin(positions, *, use_cache, draft_index):
        rope_calls.append((positions, use_cache, draft_index))
        call_index = len(rope_calls) - 1
        cos_value, sin_value = call_values[call_index]
        spec_buffers[draft_index - 1][0].fill_(cos_value)
        spec_buffers[draft_index - 1][1].fill_(sin_value)
        return spec_buffers[draft_index - 1]

    def build_local_metadata(draft_index):
        return builder._build_local_token_metadata(
            num_reqs=1,
            num_input_tokens=3,
            input_positions=input_positions,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            use_cache=True,
            draft_index=draft_index,
            local_query_start_loc=torch.zeros(2, dtype=torch.int32),
            local_seq_lens=torch.zeros(1, dtype=torch.int32),
        )

    with (
        mock.patch.object(dsa_cp, "get_tp_group", return_value=mock.Mock(world_size=1, rank_in_group=0)),
        mock.patch.object(dsa_cp, "get_cos_and_sin_dsa", side_effect=fake_get_cos_and_sin),
    ):
        draft_one = build_local_metadata(1)
        draft_two = build_local_metadata(2)
        draft_one_initial = draft_one[6].clone()
        draft_two_initial = draft_two[6].clone()
        draft_one_refresh = build_local_metadata(1)

    assert [(use_cache, draft_index) for _, use_cache, draft_index in rope_calls] == [(True, 1), (True, 2), (True, 1)]
    assert draft_one[6].data_ptr() == spec_buffers[0][0].data_ptr()
    assert draft_two[6].data_ptr() == spec_buffers[1][0].data_ptr()
    assert draft_one_refresh[6].data_ptr() == draft_one[6].data_ptr()
    assert draft_one[6].data_ptr() != draft_two[6].data_ptr()
    assert draft_one[7].data_ptr() != draft_two[7].data_ptr()
    assert torch.equal(draft_one_initial, torch.full_like(draft_one_initial, 11.0))
    assert torch.equal(draft_two_initial, torch.full_like(draft_two_initial, 22.0))
    assert torch.equal(draft_two[6], torch.full_like(draft_two[6], 22.0))
    assert torch.equal(draft_one_refresh[6], torch.full_like(draft_one_refresh[6], 33.0))


def test_build_req_metadata_for_drafting_uses_draft_rope_cache():
    builder = object.__new__(AscendDSACPMetadataBuilder)
    builder.seq_lens = torch.tensor([3], dtype=torch.int32)
    builder.seq_lens_cpu = torch.tensor([3], dtype=torch.int32)
    builder.spec_local_query_start_loc = [torch.zeros(2, dtype=torch.int32) for _ in range(2)]
    builder.spec_local_seq_lens = [torch.zeros(1, dtype=torch.int32) for _ in range(2)]
    builder.spec_start_pos = [torch.zeros(1, dtype=torch.int32) for _ in range(2)]
    builder.spec_slot_mapping = [torch.zeros(3, dtype=torch.int32) for _ in range(2)]
    builder.spec_sas_metadata = [torch.zeros(4, dtype=torch.int32) for _ in range(2)]
    builder.seqused_q = torch.tensor([], dtype=torch.int32)
    builder.block_table = torch.zeros((1, 1), dtype=torch.int32)
    builder.block_size = 128
    builder.num_actual_tokens = 3
    builder.model_config = mock.MagicMock()
    builder.model_config.get_head_size.return_value = 2
    builder.model_config.hf_config.num_attention_heads = 1
    builder.model_config.hf_config.sliding_window = 128

    local_metadata = (
        0,
        3,
        3,
        3,
        torch.tensor([0, 3], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
        torch.zeros((3, 1, 1, 2)),
        torch.zeros((3, 1, 1, 2)),
    )
    local_metadata_cpu = (*local_metadata[:6], None, None)
    builder._build_local_token_metadata = mock.MagicMock(side_effect=[local_metadata, local_metadata_cpu])

    common_attn_metadata = mock.MagicMock()
    common_attn_metadata.num_reqs = 1
    common_attn_metadata.query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    common_attn_metadata.query_start_loc_cpu = torch.tensor([0, 3], dtype=torch.int32)
    common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill

    metadata_op = mock.MagicMock(return_value=torch.ones(4, dtype=torch.int32))
    with (
        mock.patch.object(DeviceOperator, "get_dsa_sparse_attn_metadata_op", return_value=metadata_op),
        mock.patch.object(DeviceOperator, "get_dsa_sparse_attn_metadata_kwargs", return_value={}),
    ):
        builder.build_req_metadata_for_drafting(
            draft_index=2,
            common_attn_metadata=common_attn_metadata,
            input_positions=torch.arange(3),
            cos=torch.zeros((3, 1, 1, 2)),
            sin=torch.zeros((3, 1, 1, 2)),
            num_input_tokens=3,
        )

    draft_call = builder._build_local_token_metadata.call_args_list[0].kwargs
    cpu_call = builder._build_local_token_metadata.call_args_list[1].kwargs
    assert draft_call["use_cache"] is True
    assert draft_call["draft_index"] == 2
    assert cpu_call["use_cache"] is False
    assert "draft_index" not in cpu_call
