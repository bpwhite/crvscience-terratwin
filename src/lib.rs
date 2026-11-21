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

//! TerraTwin HPC Climate Modeling & Telemetry Hub (`v2.5.0`).
//! Core library crate exposing schema parsing, SQLite WAL persistence & Zarr v3 export,
//! async public climate telemetry ingestion, spherical reprojection & 3D-Var Mass-Wind Balance PCG assimilation,
//! multi-threaded Rayon climate physics simulation, and optional native `eframe`/`egui` GUI.

pub mod coastlines;
pub mod colormaps;
pub mod db;
#[cfg(feature = "gui")]
pub mod gui;
pub mod ingest;
pub mod interp;
pub mod physics;
pub mod schema;
pub mod worker;
