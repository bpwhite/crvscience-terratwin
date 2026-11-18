// Copyright (C) 2026 CRV Science Network (https://crvscience.com)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://gnu.org>.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));

    // Always emit 64x64 RGBA icon for ViewportBuilder::with_icon across all platforms.
    let rgba_64 = generate_terratwin_icon_rgba(64);
    fs::write(out_dir.join("terratwin_icon_64.rgba"), &rgba_64)
        .expect("Failed writing terratwin_icon_64.rgba");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let sizes = [16u32, 32, 48, 64, 256];
        let ico_bytes = build_multi_size_ico(&sizes);
        let ico_path = out_dir.join("terratwin.ico");
        fs::write(&ico_path, &ico_bytes).expect("Failed writing terratwin.ico");

        let ico_rc_path = ico_path.to_string_lossy().replace('\\', "/");
        let rc_content = format!(
            r#"1 ICON "{ico_rc_path}"

1 VERSIONINFO
FILEVERSION 2,5,0,0
PRODUCTVERSION 2,5,0,0
FILEFLAGSMASK 0x3fL
FILEFLAGS 0x0L
FILEOS 0x40004L
FILETYPE 0x1L
FILESUBTYPE 0x0L
BEGIN
    BLOCK "StringFileInfo"
    BEGIN
        BLOCK "040904b0"
        BEGIN
            VALUE "CompanyName", "CRV Science Network\0"
            VALUE "FileDescription", "TerraTwin HPC Climate Modeling & Telemetry Hub\0"
            VALUE "FileVersion", "2.5.0.0\0"
            VALUE "InternalName", "terratwin\0"
            VALUE "LegalCopyright", "Copyright CRV Science Network (GPL-3.0-or-later)\0"
            VALUE "OriginalFilename", "terratwin.exe\0"
            VALUE "ProductName", "TerraTwin\0"
            VALUE "ProductVersion", "2.5.0.0\0"
        END
    END
    BLOCK "VarFileInfo"
    BEGIN
        VALUE "Translation", 0x409, 1200
    END
END
"#
        );
        let rc_path = out_dir.join("terratwin.rc");
        fs::write(&rc_path, rc_content).expect("Failed writing terratwin.rc");

        let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
        if target_env == "gnu" {
            let res_obj = out_dir.join("terratwin_res.o");
            if let Some(windres) = find_windres() {
                let status = Command::new(&windres)
                    .arg(&rc_path)
                    .arg("-O")
                    .arg("coff")
                    .arg("-o")
                    .arg(&res_obj)
                    .status();
                match status {
                    Ok(s) if s.success() => {
                        println!("cargo:rustc-link-arg-bin=terratwin={}", res_obj.display());
                    }
                    Ok(s) => {
                        println!(
                            "cargo:warning=windres exited with {} — skipping Windows PE resource embedding",
                            s
                        );
                    }
                    Err(e) => {
                        println!(
                            "cargo:warning=Failed executing windres ({}): {} — skipping PE resource embedding",
                            windres, e
                        );
                    }
                }
            } else {
                println!("cargo:warning=windres not found on PATH — skipping Windows PE resource embedding");
            }
        } else if target_env == "msvc" {
            let res_file = out_dir.join("terratwin_res.res");
            if let Some(rc_exe) = find_msvc_rc() {
                let status = Command::new(&rc_exe)
                    .arg("/nologo")
                    .arg("/fo")
                    .arg(&res_file)
                    .arg(&rc_path)
                    .status();
                match status {
                    Ok(s) if s.success() => {
                        println!("cargo:rustc-link-arg-bin=terratwin={}", res_file.display());
                    }
                    Ok(s) => {
                        println!(
                            "cargo:warning=rc.exe exited with {} — skipping Windows PE resource embedding",
                            s
                        );
                    }
                    Err(e) => {
                        println!(
                            "cargo:warning=Failed executing rc.exe ({}): {} — skipping PE resource embedding",
                            rc_exe.display(),
                            e
                        );
                    }
                }
            } else {
                println!("cargo:warning=rc.exe not found — skipping Windows PE resource embedding");
            }
        }
    }
}

fn find_windres() -> Option<&'static str> {
    for candidate in ["windres", "x86_64-w64-mingw32-windres"] {
        if Command::new(candidate)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return Some(candidate);
        }
    }
    None
}

fn find_msvc_rc() -> Option<PathBuf> {
    if Command::new("rc.exe")
        .arg("/?")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return Some(PathBuf::from("rc.exe"));
    }
    let kits_root = Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
    if let Ok(entries) = fs::read_dir(kits_root) {
        let mut versions: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|v| v.path())).collect();
        versions.sort();
        versions.reverse();
        for ver_dir in versions {
            let candidate = ver_dir.join("x64").join("rc.exe");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Generate a crisp procedural TerraTwin planetary globe + atmosphere rim + climate anomaly icon
/// in unmultiplied RGBA row-major order (`size x size`).
fn generate_terratwin_icon_rgba(size: u32) -> Vec<u8> {
    let s = size as f64;
    let half = 0.5 * s;
    let r_globe = 0.40 * s;
    let r_atmo = 0.465 * s;
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);

    for py in 0..size {
        for px in 0..size {
            let x = (px as f64 + 0.5) - half;
            let y = (py as f64 + 0.5) - half;
            let dist = (x * x + y * y).sqrt();

            // Soft dark badge backdrop behind atmosphere
            let r_badge = 0.485 * s;
            if dist > r_badge + 1.0 {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
                continue;
            }

            let badge_alpha = ((r_badge + 0.8 - dist) / 1.4).clamp(0.0, 1.0);
            let mut cr = 10.0_f64;
            let mut cg = 18.0_f64;
            let mut cb = 34.0_f64;

            // Atmospheric glow halo outside globe
            if dist >= r_globe * 0.92 && dist <= r_atmo + 1.0 {
                let halo_t = (1.0 - ((dist - r_globe * 0.96).abs() / (r_atmo - r_globe * 0.88)))
                    .clamp(0.0, 1.0);
                let glow = halo_t.powf(1.6) * 0.88;
                cr = cr * (1.0 - glow) + 64.0 * glow;
                cg = cg * (1.0 - glow) + 205.0 * glow;
                cb = cb * (1.0 - glow) + 255.0 * glow;
            }

            // Inside globe sphere
            if dist <= r_globe + 0.8 {
                let nx = (x / r_globe).clamp(-1.0, 1.0);
                let ny = (-y / r_globe).clamp(-1.0, 1.0);
                let r2 = nx * nx + ny * ny;
                let nz = (1.0 - r2).max(0.0).sqrt();

                let lat = ny.asin().to_degrees();
                let lon = nx.atan2(nz.max(1e-4)).to_degrees() - 24.0;

                // Specular & Lambertian lighting from upper-left sun
                let sun = (0.45 * (-nx) + 0.55 * ny + 0.70 * nz).clamp(0.18, 1.15);

                // Synthetic stylized continents + polar ice caps + warm subtropical heat anomaly band
                let c1 = (-((lat - 32.0) / 22.0).powi(2) - ((lon + 65.0) / 26.0).powi(2)).exp();
                let c2 = (-((lat + 15.0) / 20.0).powi(2) - ((lon + 45.0) / 16.0).powi(2)).exp();
                let c3 = (-((lat - 28.0) / 24.0).powi(2) - ((lon - 18.0) / 28.0).powi(2)).exp();
                let c4 = (-((lat - 46.0) / 18.0).powi(2) - ((lon - 52.0) / 24.0).powi(2)).exp();
                let land_val = (c1 + c2 + c3 + c4)
                    + 0.14 * (lat * 0.18).sin() * (lon * 0.15).cos();

                let (mut gr, mut gg, mut gb) = if lat.abs() > 67.0 {
                    // Polar cryosphere cap
                    (222.0, 244.0, 255.0)
                } else if land_val > 0.42 {
                    // Continents: temperate green to warm arid ochre
                    let arid = (-((lat - 22.0) / 14.0).powi(2)).exp();
                    (
                        52.0 + 135.0 * arid,
                        158.0 - 20.0 * arid,
                        92.0 - 38.0 * arid,
                    )
                } else {
                    // Deep ocean + thermal anomaly gradient near equator
                    let eq_warm = (-((lat - 4.0) / 18.0).powi(2)).exp();
                    (
                        18.0 + 48.0 * eq_warm,
                        92.0 + 75.0 * eq_warm,
                        178.0 + 35.0 * eq_warm,
                    )
                };

                // Subtle graticule lines (equator + meridians)
                let eq_line = (1.0 - (lat.abs() / 4.2)).clamp(0.0, 1.0);
                let mer_dist = ((lon + 24.0) % 45.0).abs().min(45.0 - ((lon + 24.0) % 45.0).abs());
                let mer_line = (1.0 - (mer_dist / 3.2)).clamp(0.0, 1.0);
                let grid = (eq_line * 0.55 + mer_line * 0.35).min(0.65);
                gr = gr * (1.0 - grid) + 255.0 * grid;
                gg = gg * (1.0 - grid) + 215.0 * grid;
                gb = gb * (1.0 - grid) + 110.0 * grid;

                // Rim Fresnel highlight
                let rim = (1.0 - nz).powf(2.2);
                gr = (gr * sun + 75.0 * rim).min(255.0);
                gg = (gg * sun + 195.0 * rim).min(255.0);
                gb = (gb * sun + 255.0 * rim).min(255.0);

                let globe_blend = ((r_globe + 0.7 - dist) / 1.4).clamp(0.0, 1.0);
                cr = cr * (1.0 - globe_blend) + gr * globe_blend;
                cg = cg * (1.0 - globe_blend) + gg * globe_blend;
                cb = cb * (1.0 - globe_blend) + gb * globe_blend;
            }

            rgba.push(cr.clamp(0.0, 255.0) as u8);
            rgba.push(cg.clamp(0.0, 255.0) as u8);
            rgba.push(cb.clamp(0.0, 255.0) as u8);
            rgba.push((badge_alpha * 255.0).round().clamp(0.0, 255.0) as u8);
        }
    }
    rgba
}

/// Pack multiple RGBA sizes into a Windows `.ico` container using 32-bit BGRA BMP DIB entries.
fn build_multi_size_ico(sizes: &[u32]) -> Vec<u8> {
    let mut images: Vec<(u32, Vec<u8>)> = Vec::with_capacity(sizes.len());
    for &sz in sizes {
        let rgba = generate_terratwin_icon_rgba(sz);
        let dib = encode_ico_bmp_dib(sz, &rgba);
        images.push((sz, dib));
    }

    let count = images.len() as u16;
    let mut out = Vec::new();
    // ICONDIR (6 bytes)
    out.extend_from_slice(&0u16.to_le_bytes()); // Reserved
    out.extend_from_slice(&1u16.to_le_bytes()); // Type = 1 (ICON)
    out.extend_from_slice(&count.to_le_bytes());

    let mut offset = 6u32 + (images.len() as u32) * 16u32;
    for (sz, dib) in &images {
        let dim_byte = if *sz >= 256 { 0u8 } else { *sz as u8 };
        out.push(dim_byte); // Width
        out.push(dim_byte); // Height
        out.push(0); // ColorCount
        out.push(0); // Reserved
        out.extend_from_slice(&1u16.to_le_bytes()); // Planes
        out.extend_from_slice(&32u16.to_le_bytes()); // BitCount
        out.extend_from_slice(&(dib.len() as u32).to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += dib.len() as u32;
    }

    for (_, dib) in images {
        out.extend_from_slice(&dib);
    }
    out
}

fn encode_ico_bmp_dib(size: u32, rgba: &[u8]) -> Vec<u8> {
    let w = size as usize;
    let h = size as usize;
    let xor_bytes = w * h * 4;
    let mask_stride = ((w + 31) / 32) * 4;
    let and_bytes = mask_stride * h;
    let mut dib = Vec::with_capacity(40 + xor_bytes + and_bytes);

    // BITMAPINFOHEADER (40 bytes), height = 2 * h for XOR + AND masks
    dib.extend_from_slice(&40u32.to_le_bytes());
    dib.extend_from_slice(&(size as i32).to_le_bytes());
    dib.extend_from_slice(&((size * 2) as i32).to_le_bytes());
    dib.extend_from_slice(&1u16.to_le_bytes()); // Planes
    dib.extend_from_slice(&32u16.to_le_bytes()); // BitCount
    dib.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB
    dib.extend_from_slice(&(xor_bytes as u32).to_le_bytes());
    dib.extend_from_slice(&0i32.to_le_bytes());
    dib.extend_from_slice(&0i32.to_le_bytes());
    dib.extend_from_slice(&0u32.to_le_bytes());
    dib.extend_from_slice(&0u32.to_le_bytes());

    // Bottom-up BGRA pixels
    for row in (0..h).rev() {
        for col in 0..w {
            let idx = (row * w + col) * 4;
            dib.push(rgba[idx + 2]); // B
            dib.push(rgba[idx + 1]); // G
            dib.push(rgba[idx]); // R
            dib.push(rgba[idx + 3]); // A
        }
    }

    // Bottom-up 1-bpp AND mask (1 = transparent when alpha == 0)
    for row in (0..h).rev() {
        let mut mask_row = vec![0u8; mask_stride];
        for col in 0..w {
            let idx = (row * w + col) * 4;
            if rgba[idx + 3] == 0 {
                mask_row[col / 8] |= 0x80 >> (col % 8);
            }
        }
        dib.extend_from_slice(&mask_row);
    }

    dib
}
