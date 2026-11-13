//! TerraTwin HPC Climate Modeling & Telemetry Hub (`v2.4.0`).
//! Core library crate exposing schema parsing, SQLite WAL persistence,
//! async public climate telemetry ingestion, spherical reprojection & Barnes interpolation,
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
