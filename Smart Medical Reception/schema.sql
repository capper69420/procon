-- Health Monitor — Supabase / PostgreSQL schema
-- Paste this entire script into the Supabase SQL Editor and run once.

-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Patients ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS patients (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT NOT NULL,
    room            TEXT,
    age             INT,
    conditions      TEXT[] DEFAULT '{}',
    current_triage_level  CHAR(1) NOT NULL DEFAULT 'A'
        CHECK (current_triage_level IN ('A', 'B', 'C')),
    urgency_score   REAL NOT NULL DEFAULT 0.0,
    last_seen_at    TIMESTAMPTZ DEFAULT NOW(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Vision / vital readings ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS vital_readings (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id      UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    spo2            REAL,
    heart_rate      REAL,
    signal_quality  REAL,
    posture_status  TEXT,
    fall_detected   BOOLEAN DEFAULT FALSE,
    immobile_seconds REAL DEFAULT 0,
    faces_detected  INT DEFAULT 0,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vital_readings_patient_time
    ON vital_readings (patient_id, recorded_at DESC);

-- ── Audio feature readings ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audio_readings (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id      UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    breath_rate     REAL,
    cough_detected  BOOLEAN DEFAULT FALSE,
    distress_score  REAL DEFAULT 0,
    speech_clarity  REAL DEFAULT 1,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audio_readings_patient_time
    ON audio_readings (patient_id, recorded_at DESC);

-- ── Triage decision log ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS triage_events (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id          UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    triage_level        CHAR(1) NOT NULL CHECK (triage_level IN ('A', 'B', 'C')),
    urgency_score       REAL NOT NULL DEFAULT 0,
    reasons             TEXT[] DEFAULT '{}',
    recommended_action  TEXT,
    decided_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_triage_events_patient_time
    ON triage_events (patient_id, decided_at DESC);

-- ── Auto-update patients.updated_at ───────────────────────────────────────────
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS patients_updated_at ON patients;
CREATE TRIGGER patients_updated_at
    BEFORE UPDATE ON patients
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ── Enable Supabase Realtime ──────────────────────────────────────────────────
ALTER PUBLICATION supabase_realtime ADD TABLE patients;
ALTER PUBLICATION supabase_realtime ADD TABLE triage_events;
ALTER PUBLICATION supabase_realtime ADD TABLE vital_readings;

-- ── Row Level Security (permissive for hackathon demo) ────────────────────────
ALTER TABLE patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE vital_readings ENABLE ROW LEVEL SECURITY;
ALTER TABLE audio_readings ENABLE ROW LEVEL SECURITY;
ALTER TABLE triage_events ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow anon read patients" ON patients
    FOR SELECT USING (true);
CREATE POLICY "Allow anon insert patients" ON patients
    FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow anon update patients" ON patients
    FOR UPDATE USING (true);

CREATE POLICY "Allow anon read vital_readings" ON vital_readings
    FOR SELECT USING (true);
CREATE POLICY "Allow anon insert vital_readings" ON vital_readings
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow anon read audio_readings" ON audio_readings
    FOR SELECT USING (true);
CREATE POLICY "Allow anon insert audio_readings" ON audio_readings
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow anon read triage_events" ON triage_events
    FOR SELECT USING (true);
CREATE POLICY "Allow anon insert triage_events" ON triage_events
    FOR INSERT WITH CHECK (true);

-- ── Seed demo patients ────────────────────────────────────────────────────────
INSERT INTO patients (id, name, room, age, conditions, current_triage_level, urgency_score)
VALUES
    ('550e8400-e29b-41d4-a716-446655440001', 'Margaret Chen',  '101', 82, ARRAY['COPD', 'Hypertension'], 'A', 0.12),
    ('550e8400-e29b-41d4-a716-446655440002', 'Robert Williams','102', 76, ARRAY['Diabetes'],             'B', 0.55),
    ('550e8400-e29b-41d4-a716-446655440003', 'Eleanor Davis',  '103', 89, ARRAY['Heart failure'],        'C', 0.91)
ON CONFLICT (id) DO NOTHING;
