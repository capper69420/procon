import { Routes, Route } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { HomeScreen } from "./features/home/HomeScreen";
import { VoiceTriageScreen } from "./features/triage/VoiceTriageScreen";
import { PatientMonitorScreen } from "./features/monitoring/PatientMonitorScreen";
import { RoomAssignmentScreen } from "./features/assignment/RoomAssignmentScreen";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<HomeScreen />} />
        <Route path="/triage" element={<VoiceTriageScreen />} />
        <Route path="/monitor" element={<PatientMonitorScreen />} />
        <Route path="/assignment" element={<RoomAssignmentScreen />} />
      </Routes>
    </AppShell>
  );
}
