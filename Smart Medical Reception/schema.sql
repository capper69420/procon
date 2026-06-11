-- ============================================================
-- SMART MEDICAL RECEPTION KIOSK — Supabase (PostgreSQL) Schema
-- Kosen Procon | Smart Triage System v1.0
-- ============================================================
-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- 1. STAFF TABLE
--    Doctors, nurses, and admins who receive alerts/dashboards
-- ============================================================
CREATE TABLE staff (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at      TIMESTAMPTZ DEFAULT NOW(),
  name            TEXT NOT NULL,
  role            TEXT NOT NULL CHECK (role IN ('doctor', 'nurse', 'admin')),
  department      TEXT,                        -- e.g. 'emergency', 'general', 'pharmacy'
  is_on_duty      BOOLEAN DEFAULT TRUE,
  push_token      TEXT                         -- For mobile push alerts (Level C)
);

-- ============================================================
-- 2. TRIAGE SESSIONS TABLE
--    One row = one patient kiosk interaction.
--    Central linking table for all modules.
-- ============================================================
CREATE TABLE triage_sessions (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at      TIMESTAMPTZ DEFAULT NOW(),
  updated_at      TIMESTAMPTZ DEFAULT NOW(),

  -- Anonymous session token (no PII stored directly)
  session_token   TEXT UNIQUE NOT NULL DEFAULT gen_random_uuid()::TEXT,

  -- Lifecycle status (drives UI state machine)
  status          TEXT NOT NULL DEFAULT 'started'
                  CHECK (status IN (
                    'started',          -- Kiosk activated
                    'vision_complete',  -- Camera analysis done
                    'audio_complete',   -- Voice input processed
                    'triaging',         -- AI making final decision
                    'triaged',          -- Level assigned
                    'routed',           -- Patient directed to room/queue
                    'in_care',          -- Doctor opened the record
                    'discharged'        -- Session closed
                  )),

  -- Final triage outcome (populated after triaging)
  triage_level            TEXT CHECK (triage_level IN ('A', 'B', 'C')),
  routing_destination     TEXT,           -- 'emergency_room', 'waiting_b', 'pharmacy', etc.
  triage_confidence       NUMERIC(4,3),   -- 0.000 to 1.000

  -- Assigned doctor (nullable until routed)
  assigned_doctor_id      UUID REFERENCES staff(id),

  -- Override support: nurse can manually change triage level
  override_by             UUID REFERENCES staff(id),
  override_reason         TEXT,
  overridden_at           TIMESTAMPTZ
);

-- Auto-update 'updated_at' on any row change
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER triage_sessions_updated_at
  BEFORE UPDATE ON triage_sessions
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- 3. VISION ANALYSES TABLE
--    Output from MediaPipe FaceMesh + YOLOv8 pipeline
-- ============================================================
CREATE TABLE vision_analyses (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id      UUID NOT NULL REFERENCES triage_sessions(id) ON DELETE CASCADE,
  created_at      TIMESTAMPTZ DEFAULT NOW(),

  -- FaceMesh Pain Scoring (your existing algorithm output)
  pain_score      NUMERIC(4,2),           -- Normalized 0.00–10.00
  pain_score_raw  NUMERIC(5,4),           -- Raw 0.0–1.0 model output

  -- Key Action Unit (AU) values driving the pain score
  -- Example: { "AU4": 0.82, "AU6": 0.4, "AU9": 0.7, "AU43": 0.9 }
  face_action_units  JSONB,

  -- YOLO Detection Results
  -- Example: [{"class": "bleeding", "confidence": 0.94, "bbox": [x,y,w,h]},
  --           {"class": "wheelchair", "confidence": 0.88, "bbox": [...]}]
  yolo_detections    JSONB DEFAULT '[]'::JSONB,

  -- Flattened condition flags for easy SQL querying
  detected_bleeding       BOOLEAN DEFAULT FALSE,
  detected_wheelchair     BOOLEAN DEFAULT FALSE,
  detected_assisted_walk  BOOLEAN DEFAULT FALSE,
  detected_unconscious    BOOLEAN DEFAULT FALSE,
  detected_fall           BOOLEAN DEFAULT FALSE,

  -- Processing metadata
  frame_count_processed  INTEGER,
  processing_ms          INTEGER,         -- Time taken to analyze
  model_version          TEXT DEFAULT 'yolov8n+facemesh-v1'
);

-- Index for fast dashboard queries: "show all sessions with bleeding"
CREATE INDEX idx_vision_bleeding ON vision_analyses(detected_bleeding) WHERE detected_bleeding = TRUE;
CREATE INDEX idx_vision_session ON vision_analyses(session_id);

-- ============================================================
-- 4. AUDIO ANALYSES TABLE
--    Output from Whisper STT + ChatGPT EHR summarization
-- ============================================================
CREATE TABLE audio_analyses (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id      UUID NOT NULL REFERENCES triage_sessions(id) ON DELETE CASCADE,
  created_at      TIMESTAMPTZ DEFAULT NOW(),

  -- Whisper Output
  detected_language       TEXT,           -- ISO 639-1: 'ja', 'en', 'mn', 'vi', 'zh', etc.
  transcript_original     TEXT,           -- Raw Whisper output in patient's language
  transcript_english      TEXT,           -- Auto-translated to English
  audio_duration_sec      NUMERIC(6,2),
  whisper_confidence      NUMERIC(4,3),

  -- ChatGPT EHR Structuring
  ehr_summary             TEXT,           -- Full structured EHR-format summary
  chief_complaint         TEXT,           -- Single-sentence chief complaint
  symptom_duration        TEXT,           -- e.g. "since this morning", "3 days"
  pain_location           TEXT,           -- Body part mentioned: "chest", "abdomen"

  -- Structured symptom array for querying and scoring
  -- Example: ["chest_pain","shortness_of_breath","dizziness"]
  reported_symptoms       JSONB DEFAULT '[]'::JSONB,

  -- Severity keywords extracted by LLM that influenced triage
  -- Example: ["cannot breathe", "severe", "bleeding heavily"]
  severity_keywords       JSONB DEFAULT '[]'::JSONB,

  -- Audio-derived pain level (self-reported, 0–10)
  self_reported_pain      INTEGER CHECK (self_reported_pain BETWEEN 0 AND 10),

  -- LLM prompt/response audit trail (useful for contest explanation)
  llm_prompt_used         TEXT,
  llm_model_version       TEXT DEFAULT 'gpt-4o-mini',
  processing_ms           INTEGER
);

CREATE INDEX idx_audio_session ON audio_analyses(session_id);
CREATE INDEX idx_audio_language ON audio_analyses(detected_language);

-- ============================================================
-- 5. TRIAGE DECISIONS TABLE
--    The fused AI decision combining vision + audio scores.
--    Auditable: stores the exact weights used.
-- ============================================================
CREATE TABLE triage_decisions (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id      UUID NOT NULL REFERENCES triage_sessions(id) ON DELETE CASCADE,
  created_at      TIMESTAMPTZ DEFAULT NOW(),

  -- Final output
  triage_level            TEXT NOT NULL CHECK (triage_level IN ('A', 'B', 'C')),
  overall_confidence      NUMERIC(4,3),

  -- Score breakdown (for dashboard explainability)
  vision_pain_score       NUMERIC(4,2),   -- 0–10 from FaceMesh
  audio_pain_score        NUMERIC(4,2),   -- 0–10 from self-report
  vision_condition_score  NUMERIC(4,2),   -- 0–10 from YOLO flags
  audio_severity_score    NUMERIC(4,2),   -- 0–10 from LLM keyword analysis

  -- Fusion weights used in this decision
  -- Example: { "vision": 0.45, "audio": 0.55 }
  fusion_weights          JSONB,

  -- Human-readable rationale shown on Doctor's Dashboard
  -- Example: "Facial AU4/AU9 indicate severe pain (8.2/10).
  --           Patient reported chest pain for 2 hours.
  --           YOLO detected assisted walking. → LEVEL C"
  decision_rationale      TEXT,

  -- Primary trigger(s) that determined the level
  -- Example: ["bleeding_detected", "pain_score_critical", "chest_pain_keyword"]
  primary_triggers        JSONB DEFAULT '[]'::JSONB,

  -- Algorithm version for reproducibility
  algorithm_version       TEXT DEFAULT 'triage-fusion-v1'
);

CREATE INDEX idx_decision_session ON triage_decisions(session_id);
CREATE INDEX idx_decision_level ON triage_decisions(triage_level);

-- ============================================================
-- 6. QUEUE MANAGEMENT TABLE
--    Manages patient flow after triage assignment
-- ============================================================
CREATE TABLE queue_entries (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id      UUID NOT NULL REFERENCES triage_sessions(id) ON DELETE CASCADE,
  created_at      TIMESTAMPTZ DEFAULT NOW(),

  triage_level    TEXT NOT NULL CHECK (triage_level IN ('A', 'B', 'C')),
  department      TEXT NOT NULL,           -- 'emergency', 'general_1', 'pharmacy'
  queue_number    SERIAL,                  -- Auto-incrementing ticket number

  status          TEXT DEFAULT 'waiting'
                  CHECK (status IN ('waiting', 'called', 'in_progress', 'done', 'skipped')),

  -- Timing
  estimated_wait_min  INTEGER,
  enqueued_at     TIMESTAMPTZ DEFAULT NOW(),
  called_at       TIMESTAMPTZ,
  seen_at         TIMESTAMPTZ
);

CREATE INDEX idx_queue_status ON queue_entries(status, triage_level);
CREATE INDEX idx_queue_department ON queue_entries(department, status);

-- ============================================================
-- 7. ALERTS TABLE
--    Real-time push alerts sent to doctors/nurses
--    (Level C = immediate, Level B = priority queue)
-- ============================================================
CREATE TABLE alerts (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id      UUID NOT NULL REFERENCES triage_sessions(id) ON DELETE CASCADE,
  created_at      TIMESTAMPTZ DEFAULT NOW(),

  alert_type      TEXT NOT NULL
                  CHECK (alert_type IN (
                    'level_c_emergency',    -- Immediate notification
                    'level_b_priority',     -- Priority queue notification
                    'system_error',         -- AI pipeline failure
                    'override_notification' -- Manual triage override
                  )),

  target_staff_id UUID REFERENCES staff(id),
  channel         TEXT DEFAULT 'dashboard'
                  CHECK (channel IN ('dashboard', 'push', 'sms')),

  -- Alert payload shown in notification
  -- Example: { "level": "C", "reason": "Bleeding + Severe Pain (9.1/10)",
  --            "routing": "Emergency Room 2", "queue_number": 7 }
  payload         JSONB,

  sent_at         TIMESTAMPTZ DEFAULT NOW(),
  acknowledged_at TIMESTAMPTZ,
  acknowledged_by UUID REFERENCES staff(id)
);

CREATE INDEX idx_alerts_session ON alerts(session_id);
CREATE INDEX idx_alerts_unacked ON alerts(acknowledged_at) WHERE acknowledged_at IS NULL;

-- ============================================================
-- 8. SUPABASE REAL-TIME SETUP
--    Enable real-time subscriptions for the Doctor's Dashboard
-- ============================================================
-- Run these in Supabase Dashboard > Database > Replication:
-- ALTER PUBLICATION supabase_realtime ADD TABLE triage_sessions;
-- ALTER PUBLICATION supabase_realtime ADD TABLE alerts;
-- ALTER PUBLICATION supabase_realtime ADD TABLE queue_entries;

-- ============================================================
-- 9. USEFUL VIEWS FOR DOCTOR'S DASHBOARD
-- ============================================================

-- Master view: everything the doctor needs in one query
CREATE OR REPLACE VIEW active_triage_dashboard AS
SELECT
  ts.id               AS session_id,
  ts.status,
  ts.triage_level,
  ts.routing_destination,
  ts.created_at       AS arrived_at,

  -- Vision summary
  va.pain_score       AS visual_pain_score,
  va.detected_bleeding,
  va.detected_wheelchair,
  va.yolo_detections,

  -- Audio summary
  aa.detected_language,
  aa.chief_complaint,
  aa.reported_symptoms,
  aa.ehr_summary,
  aa.self_reported_pain,

  -- Decision detail
  td.decision_rationale,
  td.primary_triggers,
  td.overall_confidence,

  -- Queue info
  qe.queue_number,
  qe.department,
  qe.estimated_wait_min,

  -- Alert status
  al.alert_type,
  al.acknowledged_at  AS alert_acked_at

FROM triage_sessions ts
LEFT JOIN vision_analyses va    ON va.session_id = ts.id
LEFT JOIN audio_analyses aa     ON aa.session_id = ts.id
LEFT JOIN triage_decisions td   ON td.session_id = ts.id
LEFT JOIN queue_entries qe      ON qe.session_id = ts.id
LEFT JOIN alerts al             ON al.session_id = ts.id
WHERE ts.status NOT IN ('discharged')
ORDER BY
  CASE ts.triage_level
    WHEN 'C' THEN 1
    WHEN 'B' THEN 2
    WHEN 'A' THEN 3
    ELSE 4
  END,
  ts.created_at ASC;

-- ============================================================
-- 10. ROW-LEVEL SECURITY (RLS) — Basic setup
-- ============================================================
ALTER TABLE triage_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;

-- Doctors can read all active sessions
CREATE POLICY "staff_read_all_sessions"
  ON triage_sessions FOR SELECT
  USING (TRUE);  -- Tighten in production: USING (auth.uid() IN (SELECT id FROM staff))

-- Kiosk service role can insert sessions (use service_role key in FastAPI)
CREATE POLICY "service_insert_sessions"
  ON triage_sessions FOR INSERT
  WITH CHECK (TRUE);
