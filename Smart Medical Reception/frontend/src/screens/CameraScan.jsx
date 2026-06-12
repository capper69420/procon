import React, { useRef, useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useKiosk } from '../context/KioskContext';
import { t } from '../i18n';
import { postVision, postAudio } from '../api';

const SCAN_DURATION_MS = 8000;
const FRAME_INTERVAL_MS = 500;

export default function CameraScan() {
  const navigate = useNavigate();
  const { lang, session, updateSession, demoMode } = useKiosk();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const [cameraOk, setCameraOk] = useState(true);
  const [status, setStatus] = useState('init');
  const [progress, setProgress] = useState(0);
  const [confidence, setConfidence] = useState(0);
  const [message, setMessage] = useState('');
  const patientId = session.patient?.id;

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return null;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
  }, []);

  const runScan = useCallback(async () => {
    if (!patientId) {
      navigate('/register');
      return;
    }

    setStatus('scanning');
    setMessage(t(lang, 'scanning'));
    const start = Date.now();
    let lastVision = null;

    const tick = async () => {
      const elapsed = Date.now() - start;
      setProgress(Math.min(100, (elapsed / SCAN_DURATION_MS) * 100));

      if (cameraOk && videoRef.current) {
        const b64 = captureFrame();
        if (b64) {
          try {
            const result = await postVision(patientId, b64);
            lastVision = result;
            const sq = result.vitals?.signal_quality || 0;
            setConfidence(Math.round(sq * 100));
            setMessage(result.faces_detected > 0 ? t(lang, 'scanning') : t(lang, 'detecting'));
          } catch {
            setMessage(t(lang, 'noCamera'));
          }
        }
      }

      if (elapsed < SCAN_DURATION_MS) {
        setTimeout(tick, FRAME_INTERVAL_MS);
      } else {
        if (!lastVision) {
          lastVision = {
            vitals: { spo2: 98, heart_rate: 72, signal_quality: 0.5 },
            posture: { status: 'UNKNOWN', confidence: 0, fall_detected: false, immobile_seconds: 0 },
            faces_detected: 0,
            mock: true,
          };
        }
        try {
          await postAudio(patientId, {
            breath_rate: 16 + Math.random() * 4,
            cough_detected: false,
            distress_score: Math.max(0, Math.min(0.3, Math.random() * 0.2)),
            speech_clarity: 0.9,
          });
        } catch {
          /* audio optional */
        }
        updateSession({ vision: lastVision });
        navigate('/assessment');
      }
    };

    tick();
  }, [patientId, cameraOk, captureFrame, lang, navigate, updateSession]);

  useEffect(() => {
    if (!patientId) {
      navigate('/register');
      return undefined;
    }

    let cancelled = false;
    let stream = null;

    async function initCamera() {
      if (demoMode) {
        setCameraOk(false);
        setMessage(t(lang, 'noCamera'));
        setConfidence(85);
        setTimeout(() => { if (!cancelled) runScan(); }, 500);
        return;
      }

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((tr) => tr.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setCameraOk(true);
        setMessage(t(lang, 'detecting'));
        runScan();
      } catch {
        setCameraOk(false);
        setMessage(t(lang, 'noCamera'));
        setConfidence(75);
        runScan();
      }
    }

    initCamera();

    return () => {
      cancelled = true;
      stream?.getTracks().forEach((tr) => tr.stop());
      streamRef.current?.getTracks().forEach((tr) => tr.stop());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patientId]);

  const skipScan = () => {
    updateSession({
      vision: {
        vitals: { spo2: 97, heart_rate: 76, signal_quality: 0.6 },
        posture: { status: 'SITTING', confidence: 0.7, fall_detected: false, immobile_seconds: 0 },
        faces_detected: 1,
      },
    });
    navigate('/assessment');
  };

  return (
    <div className="screen">
      <div className="card">
        <h2 className="card-title">{t(lang, 'cameraScan')}</h2>
        <div className="status-bar">
          <span className={`status-dot ${cameraOk ? '' : 'warn'}`} />
          {t(lang, 'cameraStatus')}: {cameraOk ? 'Active' : 'Simulated'}
        </div>
        <div className="camera-container">
          <video ref={videoRef} autoPlay playsInline muted />
          <canvas ref={canvasRef} style={{ display: 'none' }} />
          {!cameraOk && (
            <div className="camera-overlay">
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '3rem' }}>📷</div>
                <p>{message}</p>
              </div>
            </div>
          )}
          {status === 'scanning' && cameraOk && (
            <div className="camera-overlay" style={{ background: 'transparent', alignItems: 'flex-end', padding: 16 }}>
              <span className="badge badge-a">{message}</span>
            </div>
          )}
        </div>
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
        <p style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
          {t(lang, 'confidence')}: {confidence}%
        </p>
        <div className="btn-row">
          <button type="button" className="btn btn-secondary" onClick={() => navigate('/register')}>
            {t(lang, 'back')}
          </button>
          <button type="button" className="btn btn-ghost" onClick={skipScan}>
            {t(lang, 'skipScan')}
          </button>
        </div>
      </div>
    </div>
  );
}
