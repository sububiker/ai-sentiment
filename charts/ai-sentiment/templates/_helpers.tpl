{{- /* Helper: chart name (may be overridden) */ -}}
{{- define "ai-sentiment.name" -}}
{{- default .Chart.Name .Values.nameOverride -}}
{{- end -}}

{{- /* Helper: full release name (handles fullnameOverride) */ -}}
{{- define "ai-sentiment.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride -}}
{{- else -}}
{{- .Release.Name -}}
{{- end -}}
{{- end -}}

{{- /* Helper: common labels (values quoted to ensure YAML strings) */ -}}
{{- define "ai-sentiment.labels" -}}
app.kubernetes.io/name: "{{ include "ai-sentiment.name" . }}"
app.kubernetes.io/instance: "{{ .Release.Name }}"
app.kubernetes.io/version: "{{ .Chart.AppVersion }}"
app.kubernetes.io/managed-by: "Helm"
{{- end -}}
