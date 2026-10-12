/**
 * DocumentAgent API client for the teacher grading mobile app.
 *
 * All requests go to the FastAPI backend at EXPO_PUBLIC_API_BASE_URL.
 * Defaults to http://localhost:8000 for local development.
 */

const BASE_URL =
  process.env.EXPO_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ExtractionMetadata {
  page_count: number;
  duration_ms: number;
  model: string;
  backend: "local" | "cloud";
}

export interface ExtractResponse {
  markdown: string;
  metadata: ExtractionMetadata;
  request_id: string;
}

export interface KeywordMatch {
  keyword: string;
  found: boolean;
  context: string | null;
}

export interface QuestionScore {
  question_id: string;
  max_marks: number;
  awarded_marks: number | null;
  keyword_matches: KeywordMatch[];
}

export interface GradeExtractResponse {
  markdown: string;
  student_label: string | null;
  keyword_matches: KeywordMatch[];
  question_scores: QuestionScore[];
  total_awarded: number | null;
  total_possible: number | null;
  metadata: Record<string, unknown>;
  request_id: string;
}

export interface RubricSummary {
  rubric_id: string;
  name: string;
  subject: string;
  grade_level: string;
  total_marks: number;
  question_count: number;
}

export interface RubricEvaluateResponse {
  rubric_id: string;
  student_label: string | null;
  question_scores: QuestionScore[];
  total_awarded: number;
  total_possible: number;
  percentage: number;
  grade: string;
  request_id: string;
}

export interface FormTemplate {
  template_id: string;
  required_fields: string[];
  optional_fields: string[];
  use_case: "admin" | "grading";
}

export interface FormFillResponse {
  fields: Record<string, string | null>;
  field_sources: Record<string, string>;
  missing_fields: string[];
  request_id: string;
}

export interface HealthResponse {
  status: string;
  local_model_available: boolean;
  cloud_api_configured: boolean;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function _handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(
      (body as { detail?: string }).detail ??
        `HTTP ${res.status}: ${res.statusText}`
    );
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/health`);
  return _handleResponse<HealthResponse>(res);
}

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

export async function extractDocument(
  fileUri: string,
  filename: string,
  opts: { useLocal?: boolean; apiKey?: string } = {}
): Promise<ExtractResponse> {
  const form = new FormData();
  form.append("file", {
    uri: fileUri,
    name: filename,
    type: "image/jpeg",
  } as unknown as Blob);
  form.append("use_local", opts.useLocal !== false ? "true" : "false");
  if (opts.apiKey) form.append("api_key", opts.apiKey);

  const res = await fetch(`${BASE_URL}/api/v1/extract`, {
    method: "POST",
    body: form,
  });
  return _handleResponse<ExtractResponse>(res);
}

// ---------------------------------------------------------------------------
// Grading
// ---------------------------------------------------------------------------

export async function gradeExtract(
  fileUri: string,
  filename: string,
  opts: {
    useLocal?: boolean;
    apiKey?: string;
    studentLabel?: string;
    keywords?: string[];
    answerKey?: object[];
  } = {}
): Promise<GradeExtractResponse> {
  const form = new FormData();
  form.append("file", {
    uri: fileUri,
    name: filename,
    type: "image/jpeg",
  } as unknown as Blob);
  form.append("use_local", opts.useLocal !== false ? "true" : "false");
  if (opts.apiKey) form.append("api_key", opts.apiKey);
  if (opts.studentLabel) form.append("student_label", opts.studentLabel);
  if (opts.keywords?.length)
    form.append("keywords", opts.keywords.join(","));
  if (opts.answerKey?.length)
    form.append("answer_key", JSON.stringify(opts.answerKey));

  const res = await fetch(`${BASE_URL}/api/v1/grade/extract`, {
    method: "POST",
    body: form,
  });
  return _handleResponse<GradeExtractResponse>(res);
}

export async function getRubrics(): Promise<RubricSummary[]> {
  const res = await fetch(`${BASE_URL}/api/v1/grade/rubrics`);
  return _handleResponse<RubricSummary[]>(res);
}

export async function evaluateRubric(
  rubricId: string,
  fileUri: string,
  filename: string,
  opts: {
    useLocal?: boolean;
    apiKey?: string;
    studentLabel?: string;
  } = {}
): Promise<RubricEvaluateResponse> {
  const form = new FormData();
  form.append("file", {
    uri: fileUri,
    name: filename,
    type: "image/jpeg",
  } as unknown as Blob);
  form.append("use_local", opts.useLocal !== false ? "true" : "false");
  if (opts.apiKey) form.append("api_key", opts.apiKey);
  if (opts.studentLabel) form.append("student_label", opts.studentLabel);

  const res = await fetch(
    `${BASE_URL}/api/v1/grade/rubrics/${rubricId}/evaluate`,
    { method: "POST", body: form }
  );
  return _handleResponse<RubricEvaluateResponse>(res);
}

// ---------------------------------------------------------------------------
// Forms
// ---------------------------------------------------------------------------

export async function getFormTemplates(): Promise<FormTemplate[]> {
  const res = await fetch(`${BASE_URL}/api/v1/forms/templates`);
  return _handleResponse<FormTemplate[]>(res);
}

export async function fillForm(
  fileUris: string[],
  filenames: string[],
  templateId: string,
  opts: { useLocal?: boolean; apiKey?: string } = {}
): Promise<FormFillResponse> {
  const form = new FormData();
  fileUris.forEach((uri, i) => {
    form.append("files", {
      uri,
      name: filenames[i],
      type: "image/jpeg",
    } as unknown as Blob);
  });
  form.append("template_id", templateId);
  form.append("use_local", opts.useLocal !== false ? "true" : "false");
  if (opts.apiKey) form.append("api_key", opts.apiKey);

  const res = await fetch(`${BASE_URL}/api/v1/forms/fill`, {
    method: "POST",
    body: form,
  });
  return _handleResponse<FormFillResponse>(res);
}
