/**
 * Zustand store for the active grading session and grade ledger.
 *
 * The grade ledger persists in Expo SQLite so marks survive app restarts.
 * The active session state lives in memory only.
 */

import { create } from "zustand";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface GradingEntry {
  id: string; // UUID
  studentLabel: string;
  subject: string;
  marksAwarded: number;
  marksPossible: number;
  percentage: number;
  grade: string;
  createdAt: string; // ISO string
  markdown: string;
}

export interface CapturedPage {
  uri: string;
  pageNumber: number;
}

export interface ActiveSession {
  studentLabel: string;
  pages: CapturedPage[];
  subject: string;
  rubricId: string | null;
  useLocal: boolean;
  apiKey: string;
}

interface GradingState {
  // Active session
  session: ActiveSession;
  setStudentLabel: (label: string) => void;
  setSubject: (subject: string) => void;
  setRubricId: (id: string | null) => void;
  setUseLocal: (value: boolean) => void;
  setApiKey: (key: string) => void;
  addPage: (page: CapturedPage) => void;
  removePage: (pageNumber: number) => void;
  reorderPages: (pages: CapturedPage[]) => void;
  clearSession: () => void;

  // Grade ledger (in-memory; also written to SQLite)
  ledger: GradingEntry[];
  addEntry: (entry: GradingEntry) => void;
  removeEntry: (id: string) => void;
  setLedger: (entries: GradingEntry[]) => void;
}

const _defaultSession: ActiveSession = {
  studentLabel: "",
  pages: [],
  subject: "",
  rubricId: null,
  useLocal: true,
  apiKey: "",
};

export const useGradingStore = create<GradingState>((set) => ({
  session: { ..._defaultSession },

  setStudentLabel: (label) =>
    set((s) => ({ session: { ...s.session, studentLabel: label } })),
  setSubject: (subject) =>
    set((s) => ({ session: { ...s.session, subject } })),
  setRubricId: (id) =>
    set((s) => ({ session: { ...s.session, rubricId: id } })),
  setUseLocal: (value) =>
    set((s) => ({ session: { ...s.session, useLocal: value } })),
  setApiKey: (key) =>
    set((s) => ({ session: { ...s.session, apiKey: key } })),

  addPage: (page) =>
    set((s) => ({
      session: {
        ...s.session,
        pages: [...s.session.pages, page],
      },
    })),
  removePage: (pageNumber) =>
    set((s) => ({
      session: {
        ...s.session,
        pages: s.session.pages
          .filter((p) => p.pageNumber !== pageNumber)
          .map((p, i) => ({ ...p, pageNumber: i + 1 })),
      },
    })),
  reorderPages: (pages) =>
    set((s) => ({ session: { ...s.session, pages } })),
  clearSession: () =>
    set({ session: { ..._defaultSession } }),

  ledger: [],
  addEntry: (entry) =>
    set((s) => ({ ledger: [entry, ...s.ledger] })),
  removeEntry: (id) =>
    set((s) => ({ ledger: s.ledger.filter((e) => e.id !== id) })),
  setLedger: (entries) => set({ ledger: entries }),
}));
