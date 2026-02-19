#!/usr/bin/env python3
"""
Log Parser for Structure Building Episodes

This script parses sender logs from multiple instances to determine:
1. Which bot (alpha or bravo) built in each episode
2. What structure type was built
3. Validates that exactly one bot builds per episode

Usage:
    python parse_structure_logs.py <input_dir>

The input_dir should contain a 'logs' subdirectory with docker-compose-* folders.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional
import json


class EpisodeInfo:
    """Container for episode information"""
    def __init__(self, episode_num: int):
        self.episode_num = episode_num
        self.alpha_structure = None
        self.alpha_builds = None  # True if building, False if observing
        self.bravo_structure = None
        self.bravo_builds = None  # True if building, False if observing
        
    def __repr__(self):
        return (f"Episode {self.episode_num}: "
                f"Alpha({self.alpha_structure}, builds={self.alpha_builds}) "
                f"Bravo({self.bravo_structure}, builds={self.bravo_builds})")
    
    def get_builder(self) -> Optional[str]:
        """Returns which bot is building: 'alpha', 'bravo', or raises an error if both bots are building"""
        if self.alpha_builds and not self.bravo_builds:
            return 'alpha'
        elif self.bravo_builds and not self.alpha_builds:
            return 'bravo'
        raise ValueError(f"Both bots are building in episode {self.episode_num}")
    
    def get_structure(self) -> Optional[str]:
        """Returns the structure that was actually built"""
        if self.alpha_builds:
            return self.alpha_structure
        elif self.bravo_builds:
            return self.bravo_structure
        return None


class LogParser:
    """Parser for sender bot logs"""

    # Regex patterns for parsing
    # Support both structureEvalPhase_0 and structureNoPlaceEvalPhase_0
    EPISODE_START_PATTERN = r'\[sendToOtherBot\] structure(NoPlace)?EvalPhase_0 beginning: Sending episode_(\d+)_structure(?:NoPlace)?EvalPhase_0'
    STRUCTURE_PATTERN = r'\[(Alpha|Bravo)\] 🎲 Randomly selected: (\S+) with (\S+)'
    # Support both "Building structure" and "[NO-PLACE] Simulating structure build..."
    BUILD_PATTERN = r'\[(Alpha|Bravo)\] 🏗️ STEP 4: (?:Building structure|\[NO-PLACE\] Simulating structure build)'
    # Support optional trailing ellipsis
    OBSERVE_PATTERN = r'\[(Alpha|Bravo)\] 🧍 STEP 4: Remaining stationary \(observer role\)\.{0,3}'

    def __init__(self, log_path: str, bot_name: str):
        self.log_path = Path(log_path)
        self.bot_name = bot_name  # 'alpha' or 'bravo'
        self.episodes: Dict[int, Dict] = {}
        self.eval_type: Optional[str] = None  # 'structureEval' or 'structureNoPlaceEval'
        
    def parse(self) -> Dict[int, Dict]:
        """Parse the log file and extract episode information"""
        if not self.log_path.exists():
            print(f"Warning: Log file not found: {self.log_path}")
            return {}
        
        with open(self.log_path, 'r') as f:
            lines = f.readlines()
        
        current_episode = None

        for line in lines:
            # Check for episode start
            episode_match = re.search(self.EPISODE_START_PATTERN, line)
            if episode_match:
                # Group 1 is "NoPlace" or None, Group 2 is episode number
                no_place_marker = episode_match.group(1)
                episode_num = int(episode_match.group(2))
                current_episode = episode_num

                # Detect eval type from the first episode found
                if self.eval_type is None:
                    self.eval_type = 'structureNoPlaceEval' if no_place_marker else 'structureEval'

                if current_episode not in self.episodes:
                    self.episodes[current_episode] = {
                        'structure': None,
                        'material': None,
                        'is_building': None
                    }
                continue
            
            if current_episode is None:
                continue
            
            # Check for structure selection (should appear soon after episode start)
            structure_match = re.search(self.STRUCTURE_PATTERN, line)
            if structure_match:
                bot = structure_match.group(1).lower()
                if bot == self.bot_name:
                    structure_type = structure_match.group(2)
                    material = structure_match.group(3)
                    self.episodes[current_episode]['structure'] = structure_type
                    self.episodes[current_episode]['material'] = material
                continue
            
            # Check for building action
            build_match = re.search(self.BUILD_PATTERN, line)
            if build_match:
                bot = build_match.group(1).lower()
                if bot == self.bot_name:
                    self.episodes[current_episode]['is_building'] = True
                continue
            
            # Check for observing action
            observe_match = re.search(self.OBSERVE_PATTERN, line)
            if observe_match:
                bot = observe_match.group(1).lower()
                if bot == self.bot_name:
                    self.episodes[current_episode]['is_building'] = False
                continue
        
        return self.episodes


class InstanceParser:
    """Parser for a complete instance (both alpha and bravo logs)"""

    def __init__(self, instance_dir: Path):
        self.instance_dir = instance_dir
        self.instance_num = self._extract_instance_num()
        self.episodes: Dict[int, EpisodeInfo] = {}
        self.eval_type: Optional[str] = None  # 'structureEval' or 'structureNoPlaceEval'
        
    def _extract_instance_num(self) -> int:
        """Extract instance number from directory name"""
        match = re.search(r'docker-compose-(\d+)', self.instance_dir.name)
        if match:
            return int(match.group(1))
        return -1
    
    def parse(self) -> Dict[int, EpisodeInfo]:
        """Parse both alpha and bravo logs for this instance"""
        alpha_log = self.instance_dir / f"sender_alpha_instance_{self.instance_num}.log"
        bravo_log = self.instance_dir / f"sender_bravo_instance_{self.instance_num}.log"

        alpha_parser = LogParser(alpha_log, 'alpha')
        bravo_parser = LogParser(bravo_log, 'bravo')

        alpha_episodes = alpha_parser.parse()
        bravo_episodes = bravo_parser.parse()

        # Capture eval type from parsers
        if alpha_parser.eval_type:
            self.eval_type = alpha_parser.eval_type
        elif bravo_parser.eval_type:
            self.eval_type = bravo_parser.eval_type

        # Merge the information
        all_episode_nums = set(alpha_episodes.keys()) | set(bravo_episodes.keys())
        
        for ep_num in all_episode_nums:
            episode_info = EpisodeInfo(ep_num)
            
            if ep_num in alpha_episodes:
                episode_info.alpha_structure = alpha_episodes[ep_num]['structure']
                episode_info.alpha_builds = alpha_episodes[ep_num]['is_building']
            
            if ep_num in bravo_episodes:
                episode_info.bravo_structure = bravo_episodes[ep_num]['structure']
                episode_info.bravo_builds = bravo_episodes[ep_num]['is_building']
            
            self.episodes[ep_num] = episode_info
        
        return self.episodes
    
    def validate(self, expected_episode_count: Optional[int] = None) -> List[str]:
        """Validate the parsed data and return list of issues"""
        issues = []

        actual_episodes = set(self.episodes.keys())

        # Check episode count if specified
        if expected_episode_count is not None:
            expected_episodes = set(range(expected_episode_count))
            if expected_episodes != actual_episodes:
                missing = expected_episodes - actual_episodes
                extra = actual_episodes - expected_episodes
                if missing:
                    issues.append(f"Instance {self.instance_num}: Missing episodes: {sorted(missing)}")
                if extra:
                    issues.append(f"Instance {self.instance_num}: Extra episodes: {sorted(extra)}")

        # Check that exactly one bot builds per episode
        for ep_num in sorted(self.episodes.keys()):
            episode = self.episodes[ep_num]
            builder = episode.get_builder()
            
            if builder is None:
                issues.append(
                    f"Instance {self.instance_num}, Episode {ep_num}: "
                    f"Cannot determine which bot is building "
                    f"(alpha_builds={episode.alpha_builds}, bravo_builds={episode.bravo_builds})"
                )
            elif episode.alpha_builds and episode.bravo_builds:
                issues.append(
                    f"Instance {self.instance_num}, Episode {ep_num}: "
                    f"Both bots are building!"
                )
            elif not episode.alpha_builds and not episode.bravo_builds:
                issues.append(
                    f"Instance {self.instance_num}, Episode {ep_num}: "
                    f"Neither bot is building!"
                )
        
        return issues
    
    def generate_report(self) -> str:
        """Generate a human-readable report"""
        lines = []
        lines.append(f"\n{'='*70}")
        lines.append(f"Instance {self.instance_num} - Structure Building Report")
        lines.append(f"{'='*70}\n")
        
        for ep_num in sorted(self.episodes.keys()):
            episode = self.episodes[ep_num]
            builder = episode.get_builder()
            structure = episode.get_structure()
            
            lines.append(f"Episode {ep_num:2d}:")
            lines.append(f"  Alpha: {episode.alpha_structure or 'N/A':15s} (builds: {episode.alpha_builds})")
            lines.append(f"  Bravo: {episode.bravo_structure or 'N/A':15s} (builds: {episode.bravo_builds})")
            
            if builder and structure:
                lines.append(f"  ✓ Builder: {builder.upper():5s} built {structure}")
            else:
                lines.append(f"  ✗ ERROR: Could not determine builder")
            lines.append("")
        
        return "\n".join(lines)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Parse structure building logs from Minecraft multiplayer evaluation'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Input directory containing logs/ subdirectory with docker-compose-* folders'
    )
    parser.add_argument(
        '--expected-episodes',
        type=int,
        default=None,
        help='Expected number of episodes per instance (for validation)'
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    logs_dir = input_dir / "logs"

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        return

    # Find all instance directories
    instance_dirs = sorted([d for d in logs_dir.iterdir()
                           if d.is_dir() and d.name.startswith('docker-compose-')])

    if not instance_dirs:
        print(f"Error: No instance directories found in {logs_dir}")
        return

    print(f"Found {len(instance_dirs)} instance(s) to parse")

    all_instances = {}
    all_issues = []
    detected_eval_type = None

    # Parse each instance
    for instance_dir in instance_dirs:
        print(f"\nParsing {instance_dir.name}...")
        inst_parser = InstanceParser(instance_dir)
        episodes = inst_parser.parse()

        if episodes:
            print(f"  Found {len(episodes)} episodes")
            all_instances[inst_parser.instance_num] = inst_parser

            # Capture eval type from first instance that has it
            if detected_eval_type is None and inst_parser.eval_type:
                detected_eval_type = inst_parser.eval_type
                print(f"  Detected eval type: {detected_eval_type}")

            # Validate
            issues = inst_parser.validate(args.expected_episodes)
            all_issues.extend(issues)

            if issues:
                print(f"  ⚠️  Found {len(issues)} validation issue(s)")
            else:
                print(f"  ✓ Validation passed")
        else:
            print(f"  ⚠️  No episodes found")

    # Print reports
    for instance_num in sorted(all_instances.keys()):
        print(all_instances[instance_num].generate_report())

    # Print validation summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}\n")

    if all_issues:
        print(f"❌ Found {len(all_issues)} issue(s):\n")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print("✅ All validations passed!")
        print(f"   - Exactly one bot builds per episode in each instance")

    # Generate JSON output
    output_data = {}
    for instance_num, inst_parser in all_instances.items():
        output_data[f"instance_{instance_num}"] = {}
        for ep_num, episode in inst_parser.episodes.items():
            output_data[f"instance_{instance_num}"][f"episode_{ep_num}"] = {
                'builder': episode.get_builder(),
                'structure': episode.get_structure(),
                'alpha_structure': episode.alpha_structure,
                'alpha_builds': episode.alpha_builds,
                'bravo_structure': episode.bravo_structure,
                'bravo_builds': episode.bravo_builds,
            }

    # Determine output filename based on eval type
    if detected_eval_type == 'structureNoPlaceEval':
        output_filename = "structure_building_no_place_summary.json"
    else:
        output_filename = "structure_building_summary.json"

    # Output to assets/hard_coded_gt/ directory (relative to this script)
    output_dir = Path(__file__).parent / "assets" / "hard_coded_gt"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n📄 Detailed data written to: {output_file}")


if __name__ == "__main__":
    main()

