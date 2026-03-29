/*
 * Fast bit packing for TLC encoder.
 * Compile: gcc -O3 -shared -fPIC -o bitpack_fast.so bitpack_fast.c
 */
#include <stdint.h>
#include <string.h>

/*
 * Pack encoded values into a uint32 bitstream (MSB-first within each word).
 *
 * entries[i]: (tier << 16) | index, or -1 for escape
 * raw_uint16[i]: original BF16 value (for escapes)
 * index_bits: bits per codebook index
 * num_values: total number of values
 * packed_out: pre-allocated uint32 buffer (must be large enough)
 * escapes_out: pre-allocated uint16 buffer for escape values
 * block_ptrs_out: pre-allocated buffer for (bit_offset, escape_index) pairs
 * block_size: values per block (typically 256)
 *
 * Returns: total bits written
 */

static inline void write_bits_fast(uint32_t *buf, uint64_t bit_pos, uint32_t value, int num_bits) {
    while (num_bits > 0) {
        int word_idx = bit_pos >> 5;
        int bit_in_word = bit_pos & 31;
        int space = 32 - bit_in_word;
        int to_write = (space < num_bits) ? space : num_bits;
        int shift = num_bits - to_write;
        uint32_t bits = (value >> shift) & ((1U << to_write) - 1);
        /* MSB-first: place bits at the top of available space */
        buf[word_idx] |= bits << (space - to_write);
        bit_pos += to_write;
        num_bits -= to_write;
    }
}

uint64_t pack_values(
    const int32_t *entries,
    const uint16_t *raw_uint16,
    int index_bits,
    int64_t num_values,
    uint32_t *packed_out,
    uint16_t *escapes_out,
    uint32_t *block_bit_offsets,
    uint32_t *block_esc_indices,
    int block_size,
    int64_t *escape_count_out
) {
    uint64_t bit_pos = 0;
    int64_t esc_idx = 0;
    int64_t block_idx = 0;

    for (int64_t i = 0; i < num_values; i++) {
        /* Record block pointer */
        if (i % block_size == 0) {
            block_bit_offsets[block_idx] = (uint32_t)bit_pos;
            block_esc_indices[block_idx] = (uint32_t)esc_idx;
            block_idx++;
        }

        int32_t entry = entries[i];

        if (entry == -1) {
            /* Escape: write 7 ones + 16 raw bits */
            write_bits_fast(packed_out, bit_pos, 0x7F, 7);
            bit_pos += 7;
            write_bits_fast(packed_out, bit_pos, (uint32_t)raw_uint16[i], 16);
            bit_pos += 16;
            escapes_out[esc_idx] = raw_uint16[i];
            esc_idx++;
        } else {
            int tier = entry >> 16;
            int index = entry & 0xFFFF;
            int prefix_bits = tier + 1;
            uint32_t prefix_val;

            if (tier == 0) {
                prefix_val = 0;  /* '0' */
            } else {
                /* tier ones followed by zero: e.g. tier=2 -> '110' */
                prefix_val = ((1U << tier) - 1) << 1;
            }

            write_bits_fast(packed_out, bit_pos, prefix_val, prefix_bits);
            bit_pos += prefix_bits;
            write_bits_fast(packed_out, bit_pos, (uint32_t)index, index_bits);
            bit_pos += index_bits;
        }
    }

    *escape_count_out = esc_idx;
    return bit_pos;
}
