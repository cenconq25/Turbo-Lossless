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
 * Pack BF16 values as 12-bit codebook indices.
 *
 * raw_uint16: input BF16 values as uint16
 * num_values: total number of values
 * reverse_map: uint32[65536] mapping value -> 12-bit index
 * packed_out: output uint32 buffer (must be pre-zeroed, size >= (num_values*12+31)/32 + 2)
 * patch_positions_out: output int32 buffer for escape positions (pre-allocated to max escapes)
 * patch_correct_out: output int16 buffer for correct BF16 values at escape positions
 * patch_wrong_out: output int16 buffer for wrong values (codebook[4095])
 * wrong_value: the int16 value at codebook[4095]
 *
 * Returns: number of escape patches
 */
/*
 * Pack BF16 values as 12-bit codebook indices with row-grouped patches (CSR format).
 *
 * Patches are sorted by row and output as:
 *   row_offsets[M+1]: CSR offsets (row_offsets[r] to row_offsets[r+1] are patches for row r)
 *   patch_cols[num_patches]: column index
 *   patch_correct[num_patches]: correct BF16 value
 *   patch_wrong[num_patches]: wrong BF16 value (codebook[4095])
 *
 * Returns: total number of patches
 */
int64_t pack_fixed12(
    const uint16_t* raw_uint16,
    int64_t num_values,
    const uint32_t* reverse_map,
    uint32_t* packed_out,
    int32_t* row_offsets_out,
    int32_t* patch_cols_out,
    int16_t* patch_correct_out,
    int16_t* patch_wrong_out,
    int16_t wrong_value,
    int32_t M,
    int32_t K)
{
    int64_t num_patches = 0;

    /* Initialize row offsets to 0 */
    for (int32_t r = 0; r <= M; r++)
        row_offsets_out[r] = 0;

    /* First pass: pack bits + count patches per row */
    for (int64_t i = 0; i < num_values; i++) {
        uint32_t idx = reverse_map[raw_uint16[i]];

        int64_t bit_pos = i * 12;
        int word = (int)(bit_pos >> 5);
        int shift = (int)(bit_pos & 31);

        packed_out[word] |= (idx & 0xFFF) << shift;
        if (shift + 12 > 32) {
            packed_out[word + 1] |= (idx & 0xFFF) >> (32 - shift);
        }

        if (idx == 4095) {
            int32_t row = (int32_t)(i / K);
            row_offsets_out[row + 1]++;
            num_patches++;
        }
    }

    /* Prefix sum to build CSR offsets */
    for (int32_t r = 0; r < M; r++)
        row_offsets_out[r + 1] += row_offsets_out[r];

    /* Second pass: fill patch data in row order */
    int32_t* tmp_offsets = (int32_t*)malloc((size_t)M * sizeof(int32_t));
    for (int32_t r = 0; r < M; r++)
        tmp_offsets[r] = row_offsets_out[r];

    for (int64_t i = 0; i < num_values; i++) {
        uint32_t idx = reverse_map[raw_uint16[i]];
        if (idx == 4095) {
            int32_t row = (int32_t)(i / K);
            int32_t col = (int32_t)(i % K);
            int32_t pos = tmp_offsets[row]++;
            patch_cols_out[pos] = col;
            patch_correct_out[pos] = (int16_t)raw_uint16[i];
            patch_wrong_out[pos] = wrong_value;
        }
    }

    free(tmp_offsets);
    return num_patches;
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
