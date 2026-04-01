/*
 * Split 12-bit packer for BF16 lossless compression.
 * Instead of contiguous 12-bit packing, splits into two byte-aligned arrays:
 *   Array 1 (sign_mantissa): 1 byte per element = [sign 1 bit][mantissa 7 bits]
 *   Array 2 (groups): 4 bits per element, nibble-packed (low=even, high=odd)
 * Total: 1.5 bytes per element (same compression ratio as 12-bit packed).
 * Decode: exponent = BaseExp + group (ONE integer add, no lookup table).
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Find the optimal BaseExp: the 15 consecutive exponents covering most weights.
 * Returns BaseExp (exponent = BaseExp + group, group 1-15).
 */
int split12_find_base_exp(const uint16_t* bf16_data, int count)
{
    int64_t counts[256] = {0};
    for (int i = 0; i < count; i++) {
        uint8_t exp = (bf16_data[i] >> 7) & 0xFF;
        counts[exp]++;
    }

    /* Find the window of 15 consecutive exponents with maximum coverage */
    int best_start = 0;
    int64_t best_sum = 0;
    for (int start = 0; start <= 241; start++) {
        int64_t sum = 0;
        for (int j = 0; j < 15; j++) sum += counts[start + j];
        if (sum > best_sum) { best_sum = sum; best_start = start; }
    }

    int base_exp = best_start - 1;  /* group 1 → exponent base_exp + 1 */
    return base_exp;
}

/*
 * Pack BF16 values into split 12-bit format with CSR escape handling.
 * sign_mantissa[count]: 1 byte per element = (sign << 7) | mantissa
 * groups[count/2]: nibble-packed, low nibble = even index, high nibble = odd index
 * When group=0 (escape), correct BF16 value stored in patch arrays.
 *
 * Returns: number of escape patches.
 */
int pack_split12(
    const uint16_t* bf16_data,
    int M,
    int K,
    int base_exp,
    uint8_t* sign_mantissa,
    uint8_t* groups,
    int32_t* row_offsets,
    int32_t* patch_cols,
    int16_t* patch_correct,
    int16_t* patch_wrong)
{
    int count = M * K;
    int num_patches = 0;

    /* Build exponent → group map */
    uint8_t exp_rmap[256];
    memset(exp_rmap, 0, 256);
    for (int g = 1; g <= 15; g++) {
        int e = base_exp + g;
        if (e >= 0 && e < 256)
            exp_rmap[e] = (uint8_t)g;
    }

    /* Clear outputs */
    memset(sign_mantissa, 0, (size_t)count);
    memset(groups, 0, (size_t)((count + 1) / 2));
    for (int r = 0; r <= M; r++)
        row_offsets[r] = 0;

    /* First pass: pack and count patches per row */
    for (int i = 0; i < count; i++) {
        uint16_t val = bf16_data[i];
        uint8_t sign = (val >> 15) & 1;
        uint8_t exp = (val >> 7) & 0xFF;
        uint8_t mantissa = val & 0x7F;
        uint8_t group = exp_rmap[exp];

        sign_mantissa[i] = (sign << 7) | mantissa;

        if (i & 1)
            groups[i / 2] |= (group << 4);
        else
            groups[i / 2] |= group;

        if (group == 0) {
            int row = i / K;
            row_offsets[row + 1]++;
            num_patches++;
        }
    }

    /* Prefix sum for row_offsets */
    for (int r = 0; r < M; r++)
        row_offsets[r + 1] += row_offsets[r];

    /* Second pass: fill patch data */
    int32_t* tmp = (int32_t*)malloc((size_t)M * sizeof(int32_t));
    for (int r = 0; r < M; r++)
        tmp[r] = row_offsets[r];

    for (int i = 0; i < count; i++) {
        uint16_t val = bf16_data[i];
        uint8_t exp = (val >> 7) & 0xFF;
        if (exp_rmap[exp] == 0) {
            int row = i / K;
            int col = i % K;
            int pos = tmp[row]++;
            patch_cols[pos] = col;
            patch_correct[pos] = (int16_t)val;
            /* patch_wrong: the incorrectly decoded value when group=0 */
            /* group=0 → exponent = base_exp + 0 = base_exp */
            uint8_t sign = (val >> 15) & 1;
            uint8_t mantissa = val & 0x7F;
            uint16_t wrong_bf16 = ((uint16_t)sign << 15) | ((uint16_t)base_exp << 7) | mantissa;
            patch_wrong[pos] = (int16_t)wrong_bf16;
        }
    }

    free(tmp);
    return num_patches;
}
