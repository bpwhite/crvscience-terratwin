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

//! Embedded vector continental coastlines, teleconnection oscillation bounding boxes,
//! and critical global telemetry / infrastructure stations for crisp zero-dependency map overlays.

/// Simplified Natural-Earth-derived continental coastline polylines `(lat_deg, lon_deg)`.
pub const CONTINENT_POLYLINES: &[&[(f64, f64)]] = &[
    // North America
    &[
        (71.3, -156.8), (70.0, -141.0), (69.5, -133.0), (68.5, -114.0), (64.0, -94.0),
        (60.0, -94.5), (53.5, -80.0), (62.0, -78.0), (60.5, -64.5), (52.0, -55.6),
        (47.0, -60.5), (44.6, -63.6), (41.5, -70.5), (35.2, -75.5), (25.2, -80.2),
        (29.0, -89.5), (26.0, -97.2), (19.2, -96.1), (21.5, -87.1), (15.8, -88.0),
        (9.0, -79.5), (15.8, -93.8), (19.0, -104.3), (23.0, -109.9), (31.3, -113.5),
        (32.5, -117.1), (37.8, -122.5), (44.6, -124.1), (48.4, -124.7), (54.7, -130.5),
        (60.0, -140.0), (61.2, -150.0), (55.0, -163.0), (60.5, -165.5), (65.6, -168.1),
        (71.3, -156.8),
    ],
    // South America
    &[
        (11.2, -74.2), (10.5, -66.9), (6.8, -58.2), (0.0, -50.0), (-5.8, -35.2),
        (-13.0, -38.5), (-23.0, -43.2), (-34.9, -56.2), (-42.8, -64.0), (-52.4, -68.4),
        (-55.0, -67.5), (-53.2, -73.0), (-41.5, -73.0), (-33.0, -71.6), (-18.5, -70.3),
        (-12.0, -77.1), (-5.0, -81.1), (1.0, -79.6), (8.9, -79.5), (11.2, -74.2),
    ],
    // Greenland
    &[
        (60.0, -44.0), (65.5, -37.0), (70.5, -22.0), (77.0, -19.0), (82.0, -24.0),
        (83.5, -38.0), (81.5, -60.0), (76.5, -69.0), (69.0, -53.0), (64.0, -51.7),
        (60.0, -44.0),
    ],
    // Europe & Eurasia
    &[
        (36.0, -5.6), (38.7, -9.4), (43.4, -8.4), (48.4, -4.5), (51.0, 1.8),
        (53.5, 8.5), (57.7, 10.6), (59.9, 10.7), (63.4, 10.4), (70.0, 20.0),
        (71.1, 27.9), (69.0, 33.1), (68.5, 44.0), (70.0, 60.0), (73.5, 80.5),
        (77.7, 104.3), (73.5, 126.0), (70.6, 160.0), (66.0, -170.0), (60.0, 165.0),
        (52.5, 158.5), (59.5, 150.8), (48.5, 140.2), (38.0, 128.8), (35.1, 129.0),
        (39.0, 121.6), (31.2, 121.5), (22.3, 114.2), (20.0, 106.0), (10.3, 107.0),
        (1.3, 103.8), (13.7, 100.5), (16.8, 96.2), (22.3, 91.8), (16.5, 82.2),
        (8.1, 77.5), (19.1, 72.9), (24.9, 67.0), (25.3, 55.3), (29.9, 48.5),
        (21.5, 39.2), (30.0, 32.5), (32.1, 34.8), (36.9, 30.7), (41.0, 29.0),
        (38.0, 23.7), (41.9, 12.5), (43.3, 5.4), (41.4, 2.2), (36.0, -5.6),
    ],
    // Africa
    &[
        (35.8, -5.8), (33.6, -7.6), (27.2, -13.2), (14.7, -17.4), (6.3, -10.8),
        (5.3, -4.0), (6.4, 3.4), (4.0, 9.7), (-4.3, 11.8), (-12.6, 13.4),
        (-22.9, 14.5), (-33.9, 18.4), (-34.8, 20.0), (-29.9, 31.0), (-25.9, 32.6),
        (-15.0, 40.7), (-6.8, 39.3), (2.0, 45.3), (11.8, 51.3), (11.6, 43.1),
        (19.6, 37.2), (29.9, 32.5), (31.2, 29.9), (32.9, 13.2), (36.8, 10.2),
        (36.8, 3.1), (35.8, -5.8),
    ],
    // Australia
    &[
        (-12.5, 130.8), (-12.2, 136.8), (-16.9, 145.8), (-27.5, 153.0), (-33.9, 151.2),
        (-37.8, 145.0), (-35.0, 138.6), (-32.0, 128.0), (-35.0, 117.9), (-32.0, 115.8),
        (-21.9, 114.1), (-18.0, 122.2), (-12.5, 130.8),
    ],
    // Antarctica
    &[
        (-63.3, -57.9), (-70.0, -61.0), (-73.5, -75.0), (-74.0, -105.0), (-74.5, -135.0),
        (-77.8, 166.7), (-71.3, 170.2), (-66.3, 140.0), (-66.6, 93.0), (-67.6, 62.9),
        (-69.0, 39.6), (-70.5, 0.0), (-75.5, -26.6), (-70.0, -60.0), (-63.3, -57.9),
    ],
];

/// Teleconnection basin bounding boxes: `(id, label, min_lat, max_lat, min_lon, max_lon)`.
pub const TELECONNECTION_BASINS: &[(&str, &str, f64, f64, f64, f64)] = &[
    ("ENSO_NINO34", "ENSO Niño 3.4", -5.0, 5.0, -170.0, -120.0),
    ("PDO_NPAC", "PDO North Pacific", 20.0, 58.0, -175.0, -125.0),
    ("NAO_NATL", "NAO North Atlantic", 36.0, 65.0, -55.0, -5.0),
    ("IOD_WEST", "IOD Western Indian", -10.0, 10.0, 50.0, 70.0),
    ("AMO_BASIN", "AMO Atlantic Multidecadal", 10.0, 60.0, -70.0, -10.0),
];

/// Major continental reference cities & polar telemetry hubs `(name, lat, lon, continent_role)`.
pub const REFERENCE_OBSERVATORIES: &[(&str, f64, f64, &str)] = &[
    // North America (3)
    ("New York City", 40.7128, -74.0061, "North America — US Atlantic Megacity"),
    ("Toronto", 43.6427, -79.3871, "North America — Great Lakes Basin"),
    ("Mexico City", 19.4333, -99.1333, "North America — High-Altitude Valley"),
    // South America (3)
    ("São Paulo", -23.5507, -46.6334, "South America — Subtropical Atlantic"),
    ("Buenos Aires", -34.6200, -58.4300, "South America — Río de la Plata"),
    ("Bogotá", 4.6097, -74.0818, "South America — Equatorial Andes"),
    // Europe (3)
    ("London", 51.5085, -0.1257, "Europe — North Atlantic Maritime"),
    ("Paris", 48.8533, 2.3489, "Europe — Western European Basin"),
    ("Berlin", 52.5200, 13.4050, "Europe — Central European Plain"),
    // Africa (3)
    ("Cairo", 30.0626, 31.2497, "Africa — Nile Delta Subtropics"),
    ("Lagos", 6.5772, 3.3211, "Africa — Gulf of Guinea Monsoon"),
    ("Nairobi", -1.2864, 36.8172, "Africa — East African Highlands"),
    // Asia (3)
    ("Tokyo", 35.6800, 139.7600, "Asia — Northwest Pacific Kuroshio"),
    ("Shanghai", 31.2300, 121.4700, "Asia — Yangtze River Delta"),
    ("Mumbai", 19.0364, 72.8595, "Asia — Arabian Sea Monsoon Coast"),
    // Oceania (3)
    ("Sydney", -33.8698, 151.2083, "Oceania — Tasman Sea Coast"),
    ("Melbourne", -37.8142, 144.9632, "Oceania — Southern Ocean Coast"),
    ("Auckland", -36.8492, 174.7653, "Oceania — Southwest Pacific"),
    // Antarctica (2)
    ("McMurdo Station", -77.8464, 166.6683, "Antarctica — Ross Island Research Hub"),
    ("Amundsen-Scott Pole", -89.9900, 0.0000, "Antarctica — Geographic South Pole"),
];
