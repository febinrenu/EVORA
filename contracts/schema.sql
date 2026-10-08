PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);          -- schema_version='1', site_name, tz, reference_now

CREATE TABLE IF NOT EXISTS cameras(
  id TEXT PRIMARY KEY,                -- 'cam_01'
  name TEXT NOT NULL,                 -- user-facing, editable
  kind TEXT NOT NULL CHECK(kind IN ('file','rtsp')),
  source_uri TEXT NOT NULL,
  source_sha256 TEXT,                 -- of the file, for evidence packs
  fps REAL, width INTEGER, height INTEGER, rotation INTEGER DEFAULT 0,
  t0 REAL NOT NULL,
  t0_source TEXT NOT NULL,            -- filename|metadata|osd|slate|manual|live
  duration_s REAL,
  site_x REAL, site_y REAL,
  status TEXT NOT NULL DEFAULT 'pending',
  layers TEXT NOT NULL DEFAULT '[]',  -- JSON list of finished layers
  ir_fraction REAL,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tracks(
  id TEXT PRIMARY KEY,                -- 'cam_01:t000123'
  camera_id TEXT NOT NULL REFERENCES cameras(id),
  cls TEXT NOT NULL, cls_conf REAL,
  t_start REAL NOT NULL, t_end REAL NOT NULL,
  n_obs INTEGER NOT NULL,
  best_crop TEXT,                     -- relative media path
  best_t REAL, best_bbox TEXT,        -- JSON [x1,y1,x2,y2]
  attrs TEXT NOT NULL DEFAULT '{}',   -- JSON, see §5.4 TrackAttrs
  direction TEXT,                     -- e.g. 'left_to_right', 'towards_camera'
  global_id TEXT,
  quality REAL
);
CREATE INDEX IF NOT EXISTS ix_tracks_cam_time ON tracks(camera_id, t_start, t_end);
CREATE INDEX IF NOT EXISTS ix_tracks_global ON tracks(global_id);

CREATE TABLE IF NOT EXISTS track_points(   -- downsampled to ~4 Hz; drives overlays and retroactive events
  track_id TEXT NOT NULL REFERENCES tracks(id),
  t REAL NOT NULL, x1 REAL, y1 REAL, x2 REAL, y2 REAL, conf REAL
);
CREATE INDEX IF NOT EXISTS ix_tp_track ON track_points(track_id, t);

CREATE TABLE IF NOT EXISTS zones(
  id TEXT PRIMARY KEY, camera_id TEXT NOT NULL REFERENCES cameras(id),
  kind TEXT NOT NULL CHECK(kind IN ('line','polygon','frame')),
  points TEXT NOT NULL DEFAULT '[]',  -- JSON [[x,y],...]
  direction TEXT NOT NULL DEFAULT 'any',
  fact_id TEXT, created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events(
  id TEXT PRIMARY KEY, camera_id TEXT NOT NULL, track_id TEXT NOT NULL,
  kind TEXT NOT NULL,                 -- cross_line|enter_zone|exit_zone|dwell|appear|disappear
  zone_id TEXT, t REAL NOT NULL, payload TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_events ON events(camera_id, kind, t);

CREATE TABLE IF NOT EXISTS global_ids(id TEXT PRIMARY KEY, cls TEXT, label TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS camera_links(
  cam_a TEXT, cam_b TEXT, mean_dt REAL, std_dt REAL, n INTEGER, overlap INTEGER DEFAULT 0,
  PRIMARY KEY(cam_a, cam_b)
);

CREATE TABLE IF NOT EXISTS memory_facts(
  id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('place','object','time')),
  canonical TEXT NOT NULL, aliases TEXT NOT NULL DEFAULT '[]',
  binding TEXT NOT NULL,              -- JSON
  source TEXT NOT NULL, confidence REAL DEFAULT 1.0,
  created_at REAL NOT NULL, last_used_at REAL, use_count INTEGER DEFAULT 0,
  superseded_by TEXT
);

CREATE TABLE IF NOT EXISTS pending_queries(   -- survives restart mid-clarification
  query_id TEXT PRIMARY KEY, text TEXT NOT NULL, plan TEXT NOT NULL,
  clarify TEXT NOT NULL, created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS plan_cache(norm_text TEXT PRIMARY KEY, plan TEXT NOT NULL, created_at REAL);
CREATE TABLE IF NOT EXISTS standing_queries(id TEXT PRIMARY KEY, text TEXT, rule TEXT, active INTEGER DEFAULT 1, created_at REAL);
CREATE TABLE IF NOT EXISTS alerts(id TEXT PRIMARY KEY, sq_id TEXT, t REAL, camera_id TEXT, track_id TEXT, evidence TEXT, acked INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS ingest_jobs(id TEXT PRIMARY KEY, camera_id TEXT, state TEXT, layer TEXT, progress REAL, rate REAL, error TEXT, updated_at REAL);
CREATE TABLE IF NOT EXISTS query_log(id TEXT PRIMARY KEY, text TEXT, plan TEXT, answer TEXT, timings TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS audit_log(id TEXT PRIMARY KEY, actor TEXT, action TEXT, detail TEXT, t REAL);
