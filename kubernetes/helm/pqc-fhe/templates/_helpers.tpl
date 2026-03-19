{{/*
PQC-FHE Helm Chart Template Helpers
=====================================
Reference: Helm Best Practices (https://helm.sh/docs/chart_best_practices/)
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "pqc-fhe.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "pqc-fhe.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "pqc-fhe.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "pqc-fhe.labels" -}}
helm.sh/chart: {{ include "pqc-fhe.chart" . }}
{{ include "pqc-fhe.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: pqc-fhe-system
{{- end }}

{{/*
Selector labels
*/}}
{{- define "pqc-fhe.selectorLabels" -}}
app.kubernetes.io/name: {{ include "pqc-fhe.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
API component labels
*/}}
{{- define "pqc-fhe.api.labels" -}}
{{ include "pqc-fhe.labels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
API selector labels
*/}}
{{- define "pqc-fhe.api.selectorLabels" -}}
{{ include "pqc-fhe.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
GPU Worker component labels
*/}}
{{- define "pqc-fhe.gpuWorker.labels" -}}
{{ include "pqc-fhe.labels" . }}
app.kubernetes.io/component: gpu-worker
{{- end }}

{{/*
GPU Worker selector labels
*/}}
{{- define "pqc-fhe.gpuWorker.selectorLabels" -}}
{{ include "pqc-fhe.selectorLabels" . }}
app.kubernetes.io/component: gpu-worker
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "pqc-fhe.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "pqc-fhe.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the Redis connection URL
*/}}
{{- define "pqc-fhe.redisUrl" -}}
{{- if .Values.redis.enabled }}
{{- if .Values.redis.auth.enabled }}
{{- printf "redis://:%s@%s-redis-master:6379" .Values.redis.auth.password (include "pqc-fhe.fullname" .) }}
{{- else }}
{{- printf "redis://%s-redis-master:6379" (include "pqc-fhe.fullname" .) }}
{{- end }}
{{- else }}
{{- .Values.externalRedis.url }}
{{- end }}
{{- end }}

{{/*
Return the appropriate apiVersion for HPA
*/}}
{{- define "pqc-fhe.hpa.apiVersion" -}}
{{- if .Capabilities.APIVersions.Has "autoscaling/v2" }}
{{- print "autoscaling/v2" }}
{{- else }}
{{- print "autoscaling/v2beta2" }}
{{- end }}
{{- end }}

{{/*
Return the appropriate apiVersion for PodDisruptionBudget
*/}}
{{- define "pqc-fhe.pdb.apiVersion" -}}
{{- if .Capabilities.APIVersions.Has "policy/v1" }}
{{- print "policy/v1" }}
{{- else }}
{{- print "policy/v1beta1" }}
{{- end }}
{{- end }}

{{/*
Return the appropriate apiVersion for NetworkPolicy
*/}}
{{- define "pqc-fhe.networkPolicy.apiVersion" -}}
{{- print "networking.k8s.io/v1" }}
{{- end }}

{{/*
Return the appropriate apiVersion for Ingress
*/}}
{{- define "pqc-fhe.ingress.apiVersion" -}}
{{- if .Capabilities.APIVersions.Has "networking.k8s.io/v1" }}
{{- print "networking.k8s.io/v1" }}
{{- else }}
{{- print "networking.k8s.io/v1beta1" }}
{{- end }}
{{- end }}

{{/*
Generate PQC algorithm configuration as JSON
Reference: NIST FIPS 203/204/205
*/}}
{{- define "pqc-fhe.pqcConfig" -}}
{
  "kem": {
    "algorithm": "{{ .Values.crypto.pqc.kemAlgorithm }}",
    "security_level": {{ .Values.crypto.pqc.kemSecurityLevel }},
    "nist_standard": "FIPS 203"
  },
  "signature": {
    "algorithm": "{{ .Values.crypto.pqc.signatureAlgorithm }}",
    "security_level": {{ .Values.crypto.pqc.signatureSecurityLevel }},
    "nist_standard": "FIPS 204"
  }
}
{{- end }}

{{/*
Generate FHE configuration as JSON
Reference: CKKS scheme parameters from DESILO FHE documentation
*/}}
{{- define "pqc-fhe.fheConfig" -}}
{
  "scheme": "CKKS",
  "poly_modulus_degree": {{ .Values.crypto.fhe.polyModulusDegree }},
  "coeff_mod_bit_sizes": {{ .Values.crypto.fhe.coeffModBitSizes | toJson }},
  "scale_bits": {{ .Values.crypto.fhe.scaleBits }},
  "security_level": {{ .Values.crypto.fhe.securityLevel }}
}
{{- end }}

{{/*
Security annotations for pods
*/}}
{{- define "pqc-fhe.securityAnnotations" -}}
seccomp.security.alpha.kubernetes.io/pod: runtime/default
container.apparmor.security.beta.kubernetes.io/{{ include "pqc-fhe.name" . }}: runtime/default
{{- end }}

{{/*
Pod security context (rootless)
Reference: Kubernetes Security Best Practices
*/}}
{{- define "pqc-fhe.podSecurityContext" -}}
runAsNonRoot: true
runAsUser: 1000
runAsGroup: 1000
fsGroup: 1000
seccompProfile:
  type: RuntimeDefault
{{- end }}

{{/*
Container security context
*/}}
{{- define "pqc-fhe.containerSecurityContext" -}}
allowPrivilegeEscalation: false
readOnlyRootFilesystem: true
capabilities:
  drop:
    - ALL
{{- end }}
