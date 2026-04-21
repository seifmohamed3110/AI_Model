# Phase 9 Manual Test Plan

Use this checklist before final demo/release.

## CV Types To Test

- [ ] Tech CV (backend or full-stack)
- [ ] Marketing CV
- [ ] Creative CV
- [ ] Weak CV (missing core sections)
- [ ] Strong CV (clear metrics and achievements)
- [ ] Student CV (limited experience)

## Expected Validation

- [ ] Detected field is correct or close to expected
- [ ] Score and grade feel fair for CV quality
- [ ] Suggestions are relevant and not generic noise
- [ ] Response contains clean structured keys:
  - `critical_issues`
  - `quick_wins`
  - `field_specific_improvements`
  - `ats_issues`
  - `writing_issues`
- [ ] Suggestions are deduplicated across sections

## Edge Cases

- [ ] Empty file upload returns clear validation error
- [ ] Very short CV returns meaningful guidance
- [ ] Weird formatting still extracts text safely
- [ ] Unsupported file type returns 400 validation message

## Calibration Notes

Use this section to tune thresholds and wording:

- Score threshold changes:
- Suggestion wording updates:
- Field-specific weighting tweaks:
