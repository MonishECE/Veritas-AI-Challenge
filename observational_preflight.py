"""Fail-closed readiness check for genuine three-method Landsat testing.

Does NOT fetch observations, invent missing weather, or assert roof accuracy.
A ready simulation is not itself successful observational validation.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

from validate_ai_landsat import check_comparison_audit

ROOT=Path(__file__).resolve().parent


def preflight(faces:Path|None,pixels:Path) -> dict:
    reasons=[]
    if not pixels.is_file():reasons.append('Verified Landsat pixel CSV missing; export actual scenes first')
    if faces is None or not faces.is_file():
        reasons.append('No new real-weather three-way thermal face CSV (physics, AI, no-reflection)')
    else:
        auditpath=faces.parent/'audit.json'
        if not auditpath.is_file():reasons.append('Thermal simulation audit.json missing')
        else:
            try:check_comparison_audit(json.loads(auditpath.read_text()))
            except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
                reasons.append(f'Thermal simulation not eligible for real-world comparison: {exc}')
    return {
        'status':'READY_FOR_EXPLORATORY_LANDSAT_COMPARISON' if not reasons else 'NOT_READY_NO_NEW_AI_OBSERVATIONAL_VALIDATION',
        'missing_or_invalid':reasons,
        'paired_AI_vs_Landsat_experiment_completed':False,
        'roof_level_field_validation_completed':False,
        'required_simulator_mode':'compare',
        'required_weather':'authentic archived weather with cryptographically matched provenance; >=24h warmup',
        'satellite_limits':'Landsat 30m gridded LST is spatially mixed; not an individual roof thermometer',
        'next_step':'python validate_ai_landsat.py --faces <REAL_COMPARE_FACE_CSV> --pixels <REAL_LANDSAT_PIXELS> --out <NEW_OUTPUT_DIR>'
        if not reasons else 'Obtain the required authentic simulation inputs before running validate_ai_landsat.py'
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--faces',type=Path,default=None)
    p.add_argument('--pixels',type=Path,default=ROOT/'urbanray_qualified_landsat_pixels_2025_2026.csv')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/geometry_ai_upgrade/observational_preflight.json')
    a=p.parse_args()
    result=preflight(a.faces,a.pixels)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
