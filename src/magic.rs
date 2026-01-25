use crate::defs::{Bitboard, EMPTY};

pub struct Magic {
    pub mask: Bitboard,
    pub magic: u64,
    pub shift: u8,
    pub offset: usize,
}

pub const BISHOP_MAGICS: [u64; 64] = [
    0x400408444040a8d, 0x2000202021000c0, 0x41010402008400, 0x820810410008,
    0x2012082000101, 0x2108500410004, 0x410080801011002, 0x8201082041001,
    0x88410214c0042, 0x1000808a044001, 0x1102008008060, 0x10100410082002,
    0x1020400081001, 0x20220040108, 0x210014202410a, 0x8201100800041,
    0x1080820820044, 0x44410100c80081, 0x1100201010e, 0x221108010010,
    0x808010010a0, 0x4002081040c010, 0x8021008401040, 0x440a02018042,
    0x2088100240081, 0x812108420010, 0x41101012010, 0x401104e010,
    0x802422002, 0x2010084000411, 0x4010a081002, 0x840104140a8,
    0x1000200a041, 0x40c00804001, 0x82200210008, 0x205041000a01,
    0x20810014022, 0x228060c008, 0x110044000a, 0x21c8121040,
    0x1204000a01, 0x204081001081, 0xc10411d, 0x4200804005,
    0x1101120040, 0x824040010, 0x8001010004, 0x100404008,
    0x40000200810, 0x201000c0108, 0x8020800402, 0x100042000404,
    0x4020000820, 0x84008001, 0x2101004010, 0x100120010,
    0x100400080208, 0x404080102, 0x10020200401, 0x800440010,
    0x800240008, 0x8008001, 0x200110020, 0x40008001
];

pub const ROOK_MAGICS: [u64; 64] = [
    0x8a80104000800020, 0x140002000100040, 0x2801880a0017001, 0x100081001000420,
    0x2000200100806001, 0x1004000800808001, 0x120020200081, 0x100040080011,
    0xc80041100200010, 0x12028100021001, 0x10008200400a01, 0x100040200100801,
    0x10082405001, 0x1000820010041, 0x1100802104, 0x10004018001101,
    0x8100040008020001, 0x400812020010010, 0x2000200100110100, 0x10002002080100,
    0x8000402001001, 0x4000210012010, 0x20012008101, 0x2000a0011041,
    0x40008002000401, 0x40001000240102, 0x81000a0802001, 0x10042200401,
    0x410008020120, 0x40008020010010, 0x40102001011, 0x4001100a04,
    0x8040210000201001, 0x40020820100040, 0x200082801000801, 0x1000842001001,
    0x10004200412, 0x100040080802, 0x1100400081, 0x11004011,
    0x21040004001001, 0x100400100201, 0x84100082001, 0x40200081001,
    0x8008004002, 0x8004008001, 0x20100408, 0x11001000401,
    0x8000508001001, 0x110008001020, 0x81000410210, 0x11000a0401,
    0x8001104, 0x801104, 0x102011, 0x1011001,
    0x2104020041001, 0x21024041012, 0x110080801001, 0x80011020020,
    0x10408010, 0x1100102, 0x21401, 0x8040102
];

pub const BISHOP_SHIFTS: [u8; 64] = [
    58, 59, 59, 59, 59, 59, 59, 58,
    59, 59, 59, 59, 59, 59, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 59, 59, 59, 59, 59, 59,
    58, 59, 59, 59, 59, 59, 59, 58
];

pub const ROOK_SHIFTS: [u8; 64] = [
    52, 53, 53, 53, 53, 53, 53, 52,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    52, 53, 53, 53, 53, 53, 53, 52
];

pub fn get_bishop_mask(sq: u8) -> Bitboard {
    let mut mask = EMPTY;
    let rank = (sq / 8) as i8;
    let file = (sq % 8) as i8;

    for (dr, df) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
        let mut r = rank + dr;
        let mut f = file + df;
        while r > 0 && r < 7 && f > 0 && f < 7 {
            mask |= 1u64 << (r * 8 + f);
            r += dr;
            f += df;
        }
    }
    mask
}

pub fn get_rook_mask(sq: u8) -> Bitboard {
    let mut mask = EMPTY;
    let rank = (sq / 8) as i8;
    let file = (sq % 8) as i8;

    for (dr, df) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let mut r = rank + dr;
        let mut f = file + df;
        while r >= 0 && r < 8 && f >= 0 && f < 8 {
            // Mask excludes the edges in the direction of movement
            if (dr != 0 && r > 0 && r < 7) || (df != 0 && f > 0 && f < 7) {
                mask |= 1u64 << (r * 8 + f);
            }
            r += dr;
            f += df;
        }
    }
    mask
}

pub fn generate_bishop_attacks_on_the_fly(sq: u8, occupancy: Bitboard) -> Bitboard {
    let mut attacks = EMPTY;
    let rank = (sq / 8) as i8;
    let file = (sq % 8) as i8;

    for (dr, df) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
        let mut r = rank + dr;
        let mut f = file + df;
        while r >= 0 && r < 8 && f >= 0 && f < 8 {
            let target = 1u64 << (r * 8 + f);
            attacks |= target;
            if (occupancy & target) != 0 { break; }
            r += dr;
            f += df;
        }
    }
    attacks
}

pub fn generate_rook_attacks_on_the_fly(sq: u8, occupancy: Bitboard) -> Bitboard {
    let mut attacks = EMPTY;
    let rank = (sq / 8) as i8;
    let file = (sq % 8) as i8;

    for (dr, df) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let mut r = rank + dr;
        let mut f = file + df;
        while r >= 0 && r < 8 && f >= 0 && f < 8 {
            let target = 1u64 << (r * 8 + f);
            attacks |= target;
            if (occupancy & target) != 0 { break; }
            r += dr;
            f += df;
        }
    }
    attacks
}

pub fn get_occupancy_from_index(index: i32, mask: Bitboard) -> Bitboard {
    let mut occupancy = EMPTY;
    let mut temp_mask = mask;
    for i in 0..mask.count_ones() {
        let sq = temp_mask.trailing_zeros() as u8;
        temp_mask &= temp_mask - 1;
        if (index & (1 << i)) != 0 {
            occupancy |= 1u64 << sq;
        }
    }
    occupancy
}
