/*
 * Fast 12-bit fixed-width packer for GPU inference format.
 * Converts BF16 values to 12-bit codebook indices and packs them.
 *
 * Compile: gcc -O3 -shared -fPIC -o fixed12_pack.so fixed12_pack.c
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Build codebook and reverse map from frequency-sorted unique values.
 *
 * sorted_vals: unique BF16 values sorted by frequency (most common first), as uint16
 * num_unique: number of unique values
 * codebook_out: output codebook[4096] as int16
 * reverse_map_out: output reverse_map[65536] as uint32 (value -> index, 4095=escape)
 *
 * Returns: number of codebook entries used (min(4095, num_unique))
 */
int build_codebook_12bit(
    const uint16_t* sorted_vals,
    int num_unique,
    int16_t* codebook_out,
    uint32_t* reverse_map_out)
{
    int cb_size = num_unique < 4095 ? num_unique : 4095;

    /* Initialize reverse map to escape sentinel */
    for (int i = 0; i < 65536; i++)
        reverse_map_out[i] = 4095;

    /* Zero codebook */
    memset(codebook_out, 0, 4096 * sizeof(int16_t));

    /* Fill codebook and reverse map */
    for (int i = 0; i < cb_size; i++) {
        codebook_out[i] = (int16_t)sorted_vals[i];
        reverse_map_out[sorted_vals[i]] = (uint32_t)i;
    }

    return cb_size;
}

/*
 * Pack with per-thread-stride escape layout for fused kernel.
 *
 * escape_offsets_out[M * workgroup_size]: per-thread offset into escape_vals
 * escape_vals_out[num_patches]: correct BF16 values, ordered by (row, tid, col_order)
 *
 * Thread tid processes columns tid, tid+ws, tid+2*ws, ... so its escapes
 * appear in encounter order in escape_vals.
 */
int64_t pack_fixed12_fused(
    const uint16_t* raw_uint16,
    int64_t num_values,
    const uint32_t* reverse_map,
    uint32_t* packed_out,
    int32_t* escape_offsets_out,
    int16_t* escape_vals_out,
    int32_t M,
    int32_t K,
    int32_t workgroup_size)
{
    int64_t num_patches = 0;
    int64_t table_size = (int64_t)M * workgroup_size;

    /* Zero the per-thread count table (reuse escape_offsets as counts first) */
    for (int64_t i = 0; i < table_size; i++)
        escape_offsets_out[i] = 0;

    /* First pass: pack 12-bit + count escapes per (row, tid) */
    for (int64_t i = 0; i < num_values; i++) {
        uint32_t idx = reverse_map[raw_uint16[i]];

        int64_t bit_pos = i * 12;
        int word = (int)(bit_pos >> 5);
        int shift = (int)(bit_pos & 31);
        packed_out[word] |= (idx & 0xFFF) << shift;
        if (shift + 12 > 32)
            packed_out[word + 1] |= (idx & 0xFFF) >> (32 - shift);

        if (idx == 4095) {
            int32_t row = (int32_t)(i / K);
            int32_t col = (int32_t)(i % K);
            int32_t tid = col % workgroup_size;
            escape_offsets_out[(int64_t)row * workgroup_size + tid]++;
            num_patches++;
        }
    }

    /* Convert counts to exclusive prefix sums (offsets) */
    int64_t running = 0;
    for (int64_t i = 0; i < table_size; i++) {
        int32_t count = escape_offsets_out[i];
        escape_offsets_out[i] = (int32_t)running;
        running += count;
    }

    /* Second pass: fill escape_vals in (row, tid, col_order) */
    int32_t* tmp_pos = (int32_t*)malloc(table_size * sizeof(int32_t));
    for (int64_t i = 0; i < table_size; i++)
        tmp_pos[i] = escape_offsets_out[i];

    for (int64_t i = 0; i < num_values; i++) {
        uint32_t idx = reverse_map[raw_uint16[i]];
        if (idx == 4095) {
            int32_t row = (int32_t)(i / K);
            int32_t col = (int32_t)(i % K);
            int32_t tid = col % workgroup_size;
            int64_t slot = (int64_t)row * workgroup_size + tid;
            escape_vals_out[tmp_pos[slot]++] = (int16_t)raw_uint16[i];
        }
    }

    free(tmp_pos);
    return num_patches;
}
