# Requirements Document

## Introduction

TI-RAN-PO-rApp is a Power Optimization rApp for Radio Access Networks. It uses reinforcement learning (via the TIBRAIN library) to learn carrier sleep/wake policies that minimize gNB power consumption while maintaining acceptable service quality. The system simulates a single gNB with 3 sectors (each having 3 frequency layers), trains an RL agent over thousands of simulated days, generates advisory carrier sleep/wake recommendations, discovers behavioural patterns from learned evidence, and presents all information through a FastAPI + HTMX web dashboard.

## Glossary

- **RAN_Environment**: The simulated Radio Access Network environment implementing the TIBRAIN Environment protocol; models a single gNB with sectors, carriers, traffic, and power
- **gNB**: A single 5G base station containing 3 sectors
- **Sector**: A directional coverage area within the gNB; each sector has 3 frequency layers (carriers)
- **Carrier**: A frequency layer within a sector; one of coverage (always ON), medium, or high capacity
- **Coverage_Carrier**: The lowest-capacity carrier in a sector that cannot be put to sleep
- **Controllable_Carrier**: A medium or high capacity carrier that may be put to sleep or woken
- **State_Encoder**: A component implementing the TIBRAIN StateEncoder protocol that discretizes RAN observations into categorical buckets
- **Power_Model**: A configurable synthetic model calculating power consumption per carrier, sector, and gNB
- **Reward_Calculator**: A component computing scalar reward from power reduction and service degradation
- **Recommendation_Engine**: A component that uses the learned TIBRAIN policy to produce advisory carrier actions with metadata
- **Discovery_Layer**: A RAN-specific interpretation layer built on TIBRAIN DiscoveryEngine that generates human-readable insights from learned patterns
- **Evaluation_Engine**: A component comparing the learned policy against an all-carriers-ON baseline across multiple metrics
- **Dashboard**: The FastAPI + HTMX web application presenting network state, recommendations, energy metrics, behaviour timelines, insights, and replay
- **Episode**: One simulated day consisting of 96 time steps at 15-minute intervals
- **Time_Step**: A single 15-minute interval within an episode
- **Training_Log**: A JSONL file recording every interaction during training
- **PRB_Utilization**: Physical Resource Block utilization as a percentage (0–100) for a carrier
- **Service_Degradation**: A condition where PRB utilization on any active carrier reaches or exceeds the configurable overload threshold
- **Legal_Actions**: The set of valid actions for a given carrier state, determined by the RAN_Environment and passed to TIBRAIN

## Requirements

### Requirement 1: RAN Environment Structure

**User Story:** As a developer, I want a simulated RAN environment with realistic topology, so that the RL agent can learn power-saving policies in a representative network structure.

#### Acceptance Criteria

1. THE RAN_Environment SHALL model exactly 1 gNB containing 3 sectors
2. THE RAN_Environment SHALL model exactly 3 carriers per sector: one Coverage_Carrier, one medium-capacity Controllable_Carrier, and one high-capacity Controllable_Carrier
3. THE RAN_Environment SHALL implement the TIBRAIN Environment protocol (reset, observe, get_legal_actions, step)
4. THE RAN_Environment SHALL operate in episodes of exactly 96 Time_Steps representing one simulated day at 15-minute intervals

### Requirement 2: Traffic Simulation

**User Story:** As a developer, I want realistic traffic patterns with daily variation, so that the agent encounters diverse load conditions during training.

#### Acceptance Criteria

1. THE RAN_Environment SHALL generate traffic per sector per carrier following a configurable daily pattern
2. THE RAN_Environment SHALL apply random variation to traffic volumes at each Time_Step
3. THE RAN_Environment SHALL assign different peak traffic times to each sector
4. WHEN a Controllable_Carrier enters sleep state, THE RAN_Environment SHALL redistribute the sleeping carrier traffic to remaining active carriers in the same sector

### Requirement 3: State Observation

**User Story:** As a developer, I want the environment to produce structured observations, so that the agent has sufficient information to make sleep/wake decisions.

#### Acceptance Criteria

1. THE RAN_Environment SHALL include time_of_day, sector_total_traffic, sector_total_load, carrier_status, DL_PRB_utilization, UL_PRB_utilization, traffic_volume, and active_UE_count in each observation
2. THE State_Encoder SHALL discretize continuous observation values into exactly five categorical buckets: VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH
3. THE State_Encoder SHALL implement the TIBRAIN StateEncoder protocol
4. THE State_Encoder SHALL be replaceable without modifying the RAN_Environment or TIBRAIN agent

### Requirement 4: Action Space

**User Story:** As a developer, I want clearly defined carrier actions with proper constraints, so that the agent only considers physically valid operations.

#### Acceptance Criteria

1. WHILE a Controllable_Carrier is ON, THE RAN_Environment SHALL present legal actions KEEP_ON and SLEEP for that carrier
2. WHILE a Controllable_Carrier is SLEEPING, THE RAN_Environment SHALL present legal actions KEEP_SLEEPING and WAKE for that carrier
3. THE RAN_Environment SHALL exclude all actions for Coverage_Carriers from the legal action set
4. THE RAN_Environment SHALL pass the legal action set to TIBRAIN via the get_legal_actions method without exposing RAN-specific semantics to TIBRAIN

### Requirement 5: Power Model

**User Story:** As a developer, I want a configurable power model, so that the system can represent different hardware profiles and calculate energy savings accurately.

#### Acceptance Criteria

1. THE Power_Model SHALL assign configurable power consumption values per carrier differentiated by ON and SLEEPING states
2. THE Power_Model SHALL compute carrier_power as the configured value for the carrier current state
3. THE Power_Model SHALL compute sector_power as the sum of carrier_power values for all carriers in a sector
4. THE Power_Model SHALL compute gNB_power as the sum of all sector_power values
5. THE Power_Model SHALL compute power_saved as the difference between all-carriers-ON baseline power and current gNB_power

### Requirement 6: Reward Calculation

**User Story:** As a developer, I want a reward function balancing energy savings and service quality, so that the agent learns policies that save power without degrading user experience.

#### Acceptance Criteria

1. THE Reward_Calculator SHALL compute reward as power_reduction_reward minus service_degradation_penalty
2. WHEN PRB_Utilization on any active carrier reaches or exceeds the configurable overload threshold, THE Reward_Calculator SHALL apply a service_degradation_penalty
3. THE RAN_Environment SHALL allow the action, simulate the resulting state, and apply the penalty rather than blocking the action
4. THE Reward_Calculator SHALL use configurable values for power_reduction_reward scaling and service_degradation_penalty magnitude
5. THE Reward_Calculator SHALL expose the overload threshold as a configurable parameter with a default value of 80 percent

### Requirement 7: TIBRAIN Integration

**User Story:** As a developer, I want the rApp to use TIBRAIN as the learning engine with no custom RL code, so that the domain layer remains cleanly separated from the learning algorithm.

#### Acceptance Criteria

1. THE RAN_Environment SHALL use tibrain.Agent as the sole learning and decision-making component
2. THE RAN_Environment SHALL use tibrain.training.train() to execute training loops with curriculum support
3. THE RAN_Environment SHALL use tibrain.persistence to save and load agent state
4. THE RAN_Environment SHALL use tibrain.discovery.DiscoveryEngine for pattern detection
5. THE RAN_Environment SHALL contain zero custom reinforcement learning algorithm code
6. THE TIBRAIN Agent SHALL receive no RAN-specific concepts (gNB, sector, carrier, PRB, UE, power, sleep, wake) through its interface

### Requirement 8: Recommendation Engine

**User Story:** As a network operator, I want advisory recommendations for each controllable carrier, so that I can make informed power-saving decisions without automated execution risk.

#### Acceptance Criteria

1. THE Recommendation_Engine SHALL produce a recommendation (KEEP, SLEEP, or WAKE) for each Controllable_Carrier based on current observations, legal actions, and the TIBRAIN learned policy
2. THE Recommendation_Engine SHALL include expected_power_reduction, service_risk, and confidence metadata with each recommendation
3. THE Recommendation_Engine SHALL persist recommendations to a recommendations.json file
4. THE Recommendation_Engine SHALL operate in advisory mode only and never auto-execute actions on the network
5. THE Dashboard SHALL display "Recommendation Only" status clearly in the user interface

### Requirement 9: Discovery and Insights

**User Story:** As a network operator, I want the system to surface learned behavioural insights, so that I can understand why certain sleep/wake patterns are beneficial.

#### Acceptance Criteria

1. THE Discovery_Layer SHALL use TIBRAIN DiscoveryEngine to detect patterns in training experience
2. THE Discovery_Layer SHALL provide a RAN-specific interpretation layer that translates generic patterns into human-readable insights referencing time, sector, carrier, traffic, PRB, action, reward, and Q-value dimensions
3. THE Discovery_Layer SHALL generate insights from learned evidence rather than using hard-coded or static text
4. THE Discovery_Layer SHALL persist discovered insights to a discoveries.json file

### Requirement 10: Evaluation

**User Story:** As a network operator, I want quantitative evaluation comparing the learned policy to a baseline, so that I can measure the value of the optimization.

#### Acceptance Criteria

1. THE Evaluation_Engine SHALL compare the learned policy against an all-carriers-ON baseline over complete episodes
2. THE Evaluation_Engine SHALL report power reduction percentage, total sleep duration, count of service degradation events, and total reward
3. THE Evaluation_Engine SHALL provide metric breakdowns by gNB, sector, carrier, and time of day
4. WHILE evaluating, THE TIBRAIN Agent SHALL operate in greedy mode without exploration

### Requirement 11: Training Infrastructure

**User Story:** As a developer, I want robust training infrastructure supporting long runs with full logging, so that the agent can learn over thousands of episodes with reproducible results.

#### Acceptance Criteria

1. THE RAN_Environment SHALL record every interaction (state, action, reward, next_state, info) in a training_log.jsonl file during training
2. THE RAN_Environment SHALL support training runs of thousands of episodes without failure
3. THE RAN_Environment SHALL use tibrain.training.train() with TrainingPhase curriculum configuration for multi-phase training

### Requirement 12: Web Dashboard

**User Story:** As a network operator, I want a web-based dashboard showing network state, recommendations, and insights, so that I can monitor and understand the rApp behaviour interactively.

#### Acceptance Criteria

1. THE Dashboard SHALL be built with FastAPI for the backend and HTMX with lightweight JavaScript for the frontend
2. THE Dashboard SHALL display Panel A: Network Overview showing gNB, sectors, and layers with status, traffic, PRB utilization, active UEs, and power per element
3. THE Dashboard SHALL display Panel B: Current Recommendations as the main panel showing per-carrier recommendations with reason text
4. THE Dashboard SHALL display Panel C: Energy View showing baseline power versus current power versus recommended power with breakdown by element
5. THE Dashboard SHALL display Panel D: 24-Hour Behaviour showing a timeline per carrier with state transitions and recommendation overlay
6. THE Dashboard SHALL display Panel E: Learned Insights presenting discovered patterns generated from evidence
7. THE Dashboard SHALL display Panel F: Replay/History allowing step-through of a previous day showing the full learning loop per time step

### Requirement 13: Demo Entry Point

**User Story:** As a developer, I want a single-command demo script, so that anyone can run the full system and see results without manual setup steps.

#### Acceptance Criteria

1. WHEN a user executes "python TI/TI-RAN-PO-rApp/experiments/demo.py", THE Demo SHALL create the gNB environment, train the agent, run evaluation, generate discoveries and recommendations, and print a summary
2. THE Demo SHALL complete the full pipeline in a single command without requiring external services or manual intervention

### Requirement 14: Domain Separation

**User Story:** As a developer, I want strict separation between RAN domain code and TIBRAIN library, so that the learning engine remains domain-agnostic and reusable.

#### Acceptance Criteria

1. THE RAN_Environment SHALL translate all RAN-specific concepts (gNB, sector, carrier, PRB, UE, power, sleep, wake) into generic states and actions before passing them to TIBRAIN
2. THE State_Encoder SHALL convert RAN observation dictionaries into opaque string keys that carry no RAN semantics recognizable by TIBRAIN
3. THE Discovery_Layer SHALL translate generic TIBRAIN patterns back into RAN-specific insights only in the interpretation layer, not within TIBRAIN itself

### Requirement 15: Project Location

**User Story:** As a developer, I want the rApp located at the correct monorepo path, so that it integrates with the existing project structure.

#### Acceptance Criteria

1. THE TI-RAN-PO-rApp source code SHALL reside at TI/TI-RAN-PO-rApp/ within the monorepo root
2. THE TI-RAN-PO-rApp SHALL use no Node.js tooling for its web stack
