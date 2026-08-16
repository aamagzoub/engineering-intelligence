# Implementation Plan: TI-RAN-PO-rApp

## Overview

Build the TI-RAN-PO-rApp power optimization rApp using Python, following a strict development order that prioritises the TIBRAIN learning loop before adding UI or advisory layers. Each task produces working code that builds on the previous step — no orphaned modules.

## Tasks

- [x] 1. Set up project structure and configuration
  - [x] 1.1 Create directory structure, README, and requirements.txt
    - Create `TI/TI-RAN-PO-rApp/` with sub-directories: `src/environment/`, `src/state/`, `src/reward/`, `src/recommendation/`, `src/discovery/`, `src/evaluation/`, `src/ran/`, `src/ui/`, `src/ui/templates/`, `src/ui/static/`, `experiments/`, `config/`, `data/`, `tests/`
    - Create `README.md` with project overview, architecture diagram, and usage instructions
    - Create `requirements.txt` listing: tibrain, fastapi, uvicorn, jinja2, python-multipart, pyyaml, hypothesis, pytest
    - Add `__init__.py` files for all Python packages
    - _Requirements: 15.1, 15.2_

  - [x] 1.2 Create `config/default.yaml` with full gNB configuration
    - Define gNB topology (3 sectors, 3 carriers each with distinct peak hours 9, 13, 18)
    - Define traffic parameters (base_load, peak_amplitude, noise_stddev, ue_per_traffic_unit)
    - Define power configuration (coverage_on, medium_on/sleeping, high_on/sleeping)
    - Define reward parameters (power_reward_scale, penalty_magnitude, overload_threshold=80)
    - Define training phases (exploration, exploitation, refinement)
    - _Requirements: 1.1, 1.2, 2.3, 5.1, 6.4, 6.5_

- [x] 2. Implement data models (Carrier, Sector, GNodeB)
  - [x] 2.1 Implement `src/environment/carrier.py`
    - Define `CarrierType` enum (COVERAGE, MEDIUM, HIGH)
    - Define `CarrierStatus` enum (ON, SLEEPING)
    - Implement `Carrier` dataclass with all fields: carrier_id, carrier_type, status, dl_prb_utilization, ul_prb_utilization, traffic_volume, active_ue_count, capacity
    - Implement `is_controllable` and `is_on` properties
    - _Requirements: 1.2, 4.1, 4.2, 4.3_

  - [x] 2.2 Implement `src/environment/sector.py`
    - Implement `Sector` dataclass with sector_id, carriers list, peak_hour
    - Implement properties: coverage_carrier, controllable_carriers, active_carriers, total_traffic, total_load
    - _Requirements: 1.1, 1.2, 2.3_

  - [x] 2.3 Implement `src/environment/gnb.py`
    - Implement `GNodeB` dataclass with gnb_id and sectors list
    - _Requirements: 1.1_

  - [ ]* 2.4 Write property tests for data model topology
    - **Property 1: Topology Invariant** — verify gNB always has 3 sectors, each with exactly 3 carriers (1 COVERAGE, 1 MEDIUM, 1 HIGH)
    - **Validates: Requirements 1.1, 1.2**

- [x] 3. Implement TrafficGenerator
  - [x] 3.1 Implement `src/environment/traffic_generator.py`
    - Implement `TrafficConfig` dataclass with base_load, peak_amplitude, noise_stddev, ue_per_traffic_unit
    - Implement `TrafficGenerator.generate(time_step, peak_hour, carrier_capacity)` returning (traffic_volume, active_ue_count)
    - Use sinusoidal daily pattern centered on peak_hour with Gaussian noise
    - Clamp traffic to [0, carrier_capacity]
    - _Requirements: 2.1, 2.2, 2.3_

  - [ ]* 3.2 Write property tests for TrafficGenerator
    - **Property 4: Traffic Stochasticity** — verify two episodes with same config produce different traffic values
    - **Property 5: Sector Peak Diversity** — verify default config has distinct peak hours per sector
    - **Validates: Requirements 2.2, 2.3**

- [x] 4. Implement PowerModel
  - [x] 4.1 Implement `src/environment/power_model.py`
    - Implement `PowerConfig` dataclass with per-carrier-type ON and SLEEPING wattage values
    - Implement `PowerModel.carrier_power(carrier)` returning configured wattage based on carrier type and status
    - Implement `PowerModel.sector_power(sector)` as sum of carrier powers
    - Implement `PowerModel.gnb_power(gnb)` as sum of sector powers
    - Implement `PowerModel.baseline_power(gnb)` computing all-ON power
    - Implement `PowerModel.power_saved(gnb)` as baseline minus current
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 4.2 Write property tests for PowerModel
    - **Property 9: Power Computation Chain** — verify carrier_power, sector_power, gnb_power, power_saved arithmetic chain holds for randomised carrier states
    - **Validates: Requirements 5.2, 5.3, 5.4, 5.5**

- [x] 5. Implement RANEnvironment (TIBRAIN Environment protocol)
  - [x] 5.1 Implement `src/environment/ran_environment.py`
    - Implement `CarrierAction` constants (KEEP_ON, SLEEP, KEEP_SLEEPING, WAKE)
    - Implement `RANEnvironment.__init__` building the gNB topology (3 sectors × 3 carriers)
    - Implement `reset()` — reset step count, set all carriers ON, generate initial traffic, return observation
    - Implement `observe()` — build observation dict with time_of_day, per-sector traffic/load, per-carrier metrics
    - Implement `get_legal_actions(state)` — return valid action pairs for controllable carriers only
    - Implement `step(action)` — apply action, advance time, regenerate traffic, redistribute sleeping carrier load, compute PRB utilization, compute reward, log interaction, return (next_state, reward, info)
    - Implement `_redistribute_traffic()` — move sleeping carrier traffic to active carriers in same sector
    - Implement `_compute_prb_utilization()` — set dl_prb/ul_prb based on traffic vs capacity
    - Implement `_log_interaction()` — append JSON line to training_log.jsonl
    - _Requirements: 1.3, 1.4, 2.4, 3.1, 4.1, 4.2, 4.3, 4.4, 6.3, 11.1_

  - [ ]* 5.2 Write property tests for RANEnvironment
    - **Property 2: Episode Length** — verify done=True at exactly step 96 and never before
    - **Property 3: Traffic Conservation on Sleep** — verify total sector traffic demand conserved when carrier sleeps
    - **Property 6: Observation Completeness** — verify all required fields present in observe() output
    - **Property 8: Legal Action Correctness** — verify legal actions satisfy ON→{KEEP_ON,SLEEP}, SLEEPING→{KEEP_SLEEPING,WAKE}, no coverage carrier actions
    - **Validates: Requirements 1.4, 2.4, 3.1, 4.1, 4.2, 4.3**

- [x] 6. Implement RANStateEncoder (TIBRAIN StateEncoder protocol)
  - [x] 6.1 Implement `src/state/ran_state_encoder.py`
    - Define `Bucket` enum (VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH)
    - Implement `RANStateEncoder.__call__(state)` encoding observation dict to opaque string key
    - Implement `_discretize(value, min_val, max_val)` mapping continuous value to one of 5 buckets using boundary thresholds [20, 40, 60, 80]
    - Key format: `"t{time_bucket}|s0:{load_bucket}{c1c2c3}|s1:...|s2:..."` — intentionally opaque to TIBRAIN
    - _Requirements: 3.2, 3.3, 3.4, 14.2_

  - [ ]* 6.2 Write property tests for RANStateEncoder
    - **Property 7: State Encoder Bucket Validity** — verify any valid value maps to exactly one of 5 buckets
    - **Property 11: Domain Separation** — verify encoded keys contain no RAN-specific terms (gNB, sector, carrier, PRB, UE, power, sleep, wake)
    - **Validates: Requirements 3.2, 7.6, 14.2**

- [x] 7. Implement RewardEngine
  - [x] 7.1 Implement `src/reward/reward_engine.py`
    - Implement `RewardEngine.__init__` with configurable power_reward_scale, penalty_magnitude, overload_threshold (default 80%)
    - Implement `compute(gnb, power_model)` returning `power_reduction_reward - penalty`
    - Implement `_compute_penalty(gnb)` applying penalty for each active carrier exceeding overload threshold
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 7.2 Write property tests for RewardEngine
    - **Property 10: Reward Formula Correctness** — verify reward = power_reduction - penalty, penalty iff PRB >= threshold, and actions are never blocked
    - **Validates: Requirements 6.1, 6.2, 6.3**

- [x] 8. Connect to TIBRAIN — first training experiment
  - [x] 8.1 Implement `experiments/train.py`
    - Import tibrain.Agent, tibrain.training.train, tibrain.training.TrainingPhase, tibrain.persistence
    - Create RANEnvironment and RANStateEncoder
    - Configure multi-phase training (exploration → exploitation → refinement)
    - Call `tibrain.training.train(agent, env, episodes, phases)` to execute training loop
    - Save trained agent via `tibrain.persistence.save(agent, path)`
    - Print training summary
    - _Requirements: 7.1, 7.2, 7.3, 7.5, 11.2, 11.3_

  - [ ]* 8.2 Write unit tests for TIBRAIN integration
    - Verify RANEnvironment conforms to TIBRAIN Environment protocol (has reset, observe, get_legal_actions, step)
    - Verify RANStateEncoder conforms to TIBRAIN StateEncoder protocol (callable returning str)
    - Run a short training loop (10 episodes) and verify no exceptions
    - **Property 17: Training Log Completeness** — verify each training step appends a complete log line
    - **Property 18: Agent Persistence Round-Trip** — verify save/load produces equivalent action selections
    - **Validates: Requirements 7.1, 7.2, 7.3, 11.1**

- [x] 9. Checkpoint — Verify core learning loop
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement Evaluator (compare against all-ON baseline)
  - [x] 10.1 Implement `src/evaluation/evaluator.py`
    - Implement `EvaluationMetrics` dataclass with power_reduction_pct, total_sleep_duration_steps, service_degradation_count, total_reward, and breakdowns by sector/carrier/time_of_day
    - Implement `Evaluator.__init__(agent, environment)`
    - Implement `evaluate(num_episodes)` — set agent to greedy mode, run complete episodes, collect per-step metrics, compute baseline comparison, aggregate results
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [ ]* 10.2 Write property tests for Evaluator
    - **Property 16: Evaluation Completeness** — verify metrics include all required top-level fields and breakdown dictionaries
    - **Validates: Requirements 10.2, 10.3**

- [x] 11. Implement RANDiscoveryAdapter
  - [x] 11.1 Implement `src/discovery/ran_discovery_adapter.py`
    - Implement `RANInsight` dataclass with summary, time_context, sector_context, carrier_context, action_context, confidence, observation_count
    - Implement `RANDiscoveryAdapter.__init__(discovery_engine)` wrapping TIBRAIN DiscoveryEngine
    - Implement `generate_insights(experiences)` — call engine.detect_patterns, then interpret each pattern into a RANInsight
    - Implement `_interpret(pattern)` mapping opaque pattern keys back to RAN dimensions (time, sector, carrier, action) using label dictionaries
    - Implement `persist(insights, path)` saving to discoveries.json
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 14.3_

  - [ ]* 11.2 Write property tests for RANDiscoveryAdapter
    - **Property 14: Discovery Insight Generation from Evidence** — verify distinct patterns produce distinct summaries referencing RAN dimensions
    - **Property 15: Discovery Persistence Round-Trip** — verify serialize/deserialize preserves all fields
    - **Validates: Requirements 9.2, 9.3, 9.4**

- [x] 12. Implement RecommendationEngine
  - [x] 12.1 Implement `src/recommendation/recommendation_engine.py`
    - Implement `Recommendation` dataclass with carrier_id, action (KEEP/SLEEP/WAKE), expected_power_reduction, service_risk, confidence, reason
    - Implement `RecommendationEngine.__init__(agent, environment)`
    - Implement `generate()` — set agent to greedy mode, observe state, get legal actions, query policy per carrier, build recommendations with metadata
    - Implement `persist(recommendations, path)` saving to recommendations.json
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [ ]* 12.2 Write property tests for RecommendationEngine
    - **Property 12: Recommendation Validity** — verify each recommendation references valid controllable carrier, has valid action, includes all metadata
    - **Property 13: Recommendation Persistence Round-Trip** — verify serialize/deserialize preserves all fields
    - **Validates: Requirements 8.1, 8.2, 8.3**

- [x] 13. Checkpoint — Verify full pipeline (train + evaluate + discover + recommend)
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Implement Dashboard (FastAPI + HTMX)
  - [x] 14.1 Implement `src/ui/app.py` with FastAPI application
    - Create FastAPI app with Jinja2Templates and StaticFiles setup
    - Implement `GET /` serving the main dashboard HTML page
    - Implement `GET /api/network-overview` returning Panel A data (gNB, sectors, carriers with status/traffic/PRB/UEs/power)
    - Implement `GET /api/recommendations` returning Panel B data from recommendations.json
    - Implement `GET /api/energy` returning Panel C data (baseline vs current vs recommended power)
    - Implement `GET /api/behaviour-timeline` returning Panel D data (24h carrier state transitions)
    - Implement `GET /api/insights` returning Panel E data from discoveries.json
    - Implement `GET /api/replay/{step}` returning Panel F data for a specific time step from training_log.jsonl
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 15.2_

  - [x] 14.2 Create HTMX templates and static assets
    - Create `src/ui/templates/index.html` — base layout with all 6 panels, HTMX partials loading, "Recommendation Only" status badge
    - Create `src/ui/templates/partials/` — individual HTMX partial templates for each panel (network_overview.html, recommendations.html, energy.html, timeline.html, insights.html, replay.html)
    - Create `src/ui/static/style.css` — minimal responsive styling
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 8.5_

  - [ ]* 14.3 Write unit tests for Dashboard endpoints
    - Test each API endpoint returns valid JSON/HTML
    - Test graceful handling of missing data files
    - Test "Recommendation Only" status displayed
    - _Requirements: 12.1, 8.5_

- [x] 15. Implement RAN Measurements module
  - [x] 15.1 Implement `src/ran/measurements.py` and `src/ran/measurement_mapper.py`
    - Implement `RANMeasurement` dataclass
    - Implement `MeasurementMapper.extract_measurements(gnb, time_step)` returning list of RANMeasurement for all carriers
    - _Requirements: 3.1_

- [x] 16. Implement `experiments/demo.py` (end-to-end)
  - [x] 16.1 Implement `experiments/demo.py`
    - Create RANEnvironment, RANStateEncoder, and TIBRAIN Agent
    - Run short training (500 episodes, 2 phases)
    - Save model via tibrain.persistence
    - Run evaluation (5 episodes)
    - Run discovery to generate insights
    - Generate recommendations
    - Print summary (episodes, power reduction %, insights count, recommendations count)
    - Ensure single-command execution: `python TI/TI-RAN-PO-rApp/experiments/demo.py`
    - _Requirements: 13.1, 13.2_

  - [ ]* 16.2 Write smoke test for demo.py
    - Verify demo.py runs end-to-end without error
    - Verify all expected output files are created (q_table.json, discoveries.json, recommendations.json, training_log.jsonl)
    - _Requirements: 13.1, 13.2_

- [x] 17. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- The learning loop (tasks 1–9) is fully functional before any UI work begins
- Each task references specific requirements for traceability
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The project uses Python exclusively — no Node.js tooling for the web stack (FastAPI + HTMX)
- All code resides at `TI/TI-RAN-PO-rApp/` within the monorepo root

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "2.2", "2.3"] },
    { "id": 2, "tasks": ["2.4", "3.1", "4.1"] },
    { "id": 3, "tasks": ["3.2", "4.2", "5.1"] },
    { "id": 4, "tasks": ["5.2", "6.1", "7.1"] },
    { "id": 5, "tasks": ["6.2", "7.2", "8.1"] },
    { "id": 6, "tasks": ["8.2"] },
    { "id": 7, "tasks": ["10.1", "11.1", "12.1", "15.1"] },
    { "id": 8, "tasks": ["10.2", "11.2", "12.2"] },
    { "id": 9, "tasks": ["14.1"] },
    { "id": 10, "tasks": ["14.2", "14.3"] },
    { "id": 11, "tasks": ["16.1"] },
    { "id": 12, "tasks": ["16.2"] }
  ]
}
```
