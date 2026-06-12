import { createClient } from '@supabase/supabase-js';

const url = import.meta.env.VITE_SUPABASE_URL || '';
const key = import.meta.env.VITE_SUPABASE_ANON_KEY || '';

export const supabaseConfigured = Boolean(url && key);

export const supabase = supabaseConfigured
  ? createClient(url, key, { realtime: { params: { eventsPerSecond: 5 } } })
  : null;

export function subscribePatients(onUpdate) {
  if (!supabase) return () => {};

  const channel = supabase
    .channel('patients-changes')
    .on('postgres_changes', { event: '*', schema: 'public', table: 'patients' }, () => {
      onUpdate();
    })
    .on('postgres_changes', { event: '*', schema: 'public', table: 'triage_events' }, () => {
      onUpdate();
    })
    .subscribe((status) => {
      if (status === 'CHANNEL_ERROR' || status === 'TIMED_OUT') {
        onUpdate({ offline: true });
      }
    });

  return () => {
    supabase.removeChannel(channel);
  };
}
