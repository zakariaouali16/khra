import os
import json
import pandas as pd
import requests
from tqdm import tqdm


class LLMRouterV2:

    # ── Safety keywords for pre-gate override ─────────────────────────────────
    # ONLY physical emergencies and psychiatric crises belong here.
    # Interpersonal safety / harassment / feeling socially unsafe are NOT
    # physical emergencies — the LLM router handles those via the
    # Instructor/TA Escalation definition.
    SAFETY_KEYWORDS = [
        # suicidal ideation
        "suicide", "kill myself", "end my life", "want to die",
        "no point in going on", "there's no point anymore", "can't go on",
        "don't want to be here anymore", "thinking about ending it",
        "feel like ending it",
        # self-harm
        "self-harm", "self harm", "cutting myself", "hurt myself",
        "going to hurt myself", "harming myself", "thoughts of hurting",
        "telling me to hurt",
        # cardiac / stroke signals
        "chest is tight", "chest feels tight", "chest feels like",
        "chest pain", "crushing chest pain",
        "arm is numb", "having a stroke", "face is drooping",
        "severe chest pain", "shortness of breath",
        # breathing
        "can't breathe", "cannot breathe", "can barely breathe",
        "not breathing", "stopped breathing",
        # throat / allergic reaction
        "throat is swelling", "throat feels like it's closing",
        "throat feels like it's swelling shut", "allergic reaction",
        "throat is starting to swell",
        # unconscious / unresponsive
        "passed out", "fainted", "collapsed", "unresponsive",
        "won't wake up", "not waking up", "collapsed on the floor",
        "not responding", "isn't waking up",
        # bleeding
        "bleed", "won't stop bleeding", "severe bleeding",
        "coughing up blood", "bleeding won't stop",
        "soaking through the towels",
        # medication / poisoning
        "overdose", "took too many pills", "took too much",
        "took more than", "took more medication", "accidentally took",
        "swallowed a whole", "swallowed something",
        "drank a cleaning chemical", "drank half a bottle",
        "feel strange after taking", "feel weird after taking",
        "feel sick after taking", "took the wrong",
        "double my dose",
        # vision / neurological
        "lost all vision", "lost vision", "vision is going dark",
        "losing consciousness", "seizure", "having a seizure",
        # choking
        "choking", "making a choking sound",
        # acute physical distress
        "skin is peeling off", "thunderclap headache",
        "shaking and can't stop", "high fever", "burning up",
        "fever won't break", "hit head", "broken bone", "deep cut",
        # obstetric emergency
        "water just broke", "water broke",
        # psychiatric emergency
        "voices telling me to hurt", "can't fight them off",
        # injury / trauma
        "cannot feel my legs", "can't feel my legs",
        "fell off the roof",
    ]

    # ── Academic safety keywords for Instructor/TA escalation override ─────────
    # Covers: active exam cheating attempts and academic integrity violations.
    # These are deterministic rules — the LLM should not decide on these.
    ACADEMIC_EXAM_PHRASES = [
        "in the middle of the exam", "in the middle of my exam",
        "in the middle of the midterm", "in the middle of my midterm",
        "in the middle of the test", "in the middle of my test",
        "in the middle of the final", "in the middle of my final",
        "in the middle of the quiz", "in the middle of my quiz",
        "currently taking the exam", "currently taking my exam",
        "currently taking the midterm", "currently taking the test",
        "currently taking the quiz", "currently taking the final",
        "taking the exam right now", "taking my exam right now",
        "taking the midterm right now", "taking the test right now",
        "taking the quiz right now", "taking the final right now",
        "doing the exam right now", "doing my exam right now",
        "doing the midterm right now", "doing the test right now",
        "right now. quick", "right now, quick",
        "online midterm right now", "online exam right now",
        "online final right now", "online quiz right now",
        "online test right now",
    ]

    ACADEMIC_INTEGRITY_KEYWORDS = [
        "plagiarism", "plagiarized", "plagiarizing",
        "academic dishonesty", "academic integrity violation",
        "cheating accusation", "accused of cheating",
        "accused of plagiarism", "flagged for plagiarism",
        "flagged for cheating", "academic misconduct",
        "integrity violation",
    ]

    def __init__(self, model_name='Gemma4', taxonomy_path='../../Data/taxonomy_phase5.json'):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

        print(f"Loading taxonomy from {taxonomy_path}...")
        with open(taxonomy_path, 'r') as f:
            self.taxonomy = json.load(f)

        # Build valid label set at init time — includes gating outcomes + all domain labels.
        # Used by the hallucination guard to catch any invented label names.
        self.valid_labels = {
            label['name']
            for domain_data in self.taxonomy['domains'].values()
            for label in domain_data['labels']
        }

    # ── Pre-gate safety check ──────────────────────────────────────────────────
    def _check_safety_override(self, user_prompt):
        """
        Deterministic keyword scan run BEFORE the gate or router.
        Only fires on unambiguous physical emergencies and psychiatric crises.
        Returns "Urgent Escalation" if matched, None otherwise.
        """
        prompt_lower = user_prompt.lower()
        for keyword in self.SAFETY_KEYWORDS:
            if keyword in prompt_lower:
                return "Urgent Escalation"
        return None

    # ── Pre-gate academic safety check ────────────────────────────────────────
    def _check_academic_override(self, user_prompt):
        """
        Deterministic keyword scan for academic integrity violations and
        active exam cheating attempts. Both must route to Instructor/TA Escalation
        regardless of how the request is framed.
        Returns "Instructor/TA Escalation" if matched, None otherwise.
        """
        prompt_lower = user_prompt.lower()
        for phrase in self.ACADEMIC_EXAM_PHRASES:
            if phrase in prompt_lower:
                return "Instructor/TA Escalation"
        for keyword in self.ACADEMIC_INTEGRITY_KEYWORDS:
            if keyword in prompt_lower:
                return "Instructor/TA Escalation"
        return None

    # ── Hallucination guard ────────────────────────────────────────────────────
    def _validate_label(self, predicted):
        """
        Ensures the LLM's predicted label exists in the taxonomy.

        Resolution order:
          1. Exact match             → return as-is
          2. Case-insensitive match  → return canonical casing
          3. Substring match         → return canonical label
          4. No match                → fall back to "Clarification Needed"

        Returns (validated_label, was_corrected).
        """
        if predicted in self.valid_labels:
            return predicted, False

        predicted_lower = predicted.strip().lower()

        # Case-insensitive exact match
        for valid in self.valid_labels:
            if valid.lower() == predicted_lower:
                return valid, True

        # Substring match (handles truncated / paraphrased label names)
        for valid in self.valid_labels:
            if valid.lower() in predicted_lower or predicted_lower in valid.lower():
                return valid, True

        # No match — fall back to Clarification Needed
        tqdm.write(f"  [hallucination guard] '{predicted}' not in taxonomy → Clarification Needed")
        return "Clarification Needed", True

    def build_system_prompt(self):
        """
        Constructs the routing prompt with gating outcomes first,
        then education and healthcare domain labels, with an explicit
        domain separation rule to prevent cross-domain confusion.
        """
        gating_text = ""
        domain_text = ""

        for domain_name, domain_data in self.taxonomy['domains'].items():
            section = f"\n### {domain_name.upper().replace('_', ' ')} ###\n"
            for l in domain_data['labels']:
                slots = ", ".join(l.get('required_slots', [])) if l.get('required_slots') else "None"
                section += f"- {l['name']}: {l['definition']} (Required slots: {slots})\n"

            if domain_name == 'gating_outcomes':
                gating_text += section
            else:
                domain_text += section

        # Build exhaustive allowed-label list dynamically from taxonomy
        allowed_labels = sorted(self.valid_labels)
        allowed_labels_text = "\n".join(f'  - "{name}"' for name in allowed_labels)

        system_prompt = f"""You are an expert, autonomous routing agent.

STEP 1 — Check gating outcomes FIRST. If any apply, route there immediately and do not consider domain labels:
{gating_text}

STEP 2 — Only if no gating outcome applies, route to exactly one domain label:

CRITICAL DOMAIN SEPARATION RULE: Education labels (Concept Explanation, Debugging & Code Troubleshooting, Assignment & Grading Policy, Exam & Assessment Prep, Course Logistics & Environment Setup, Instructor/TA Escalation) apply ONLY to computing/programming student support. Healthcare labels (Scheduling & Appointments, Insurance & Billing, Facility & General Information, Pharmacy & Prescription Logistics, Human Staff Review Needed) apply ONLY to non-clinical patient/administrative support. NEVER route a healthcare question to an education label or vice versa. "Course Logistics & Environment Setup" is an EDUCATION-ONLY label — do NOT use it for hospital, clinic, or medical facility questions about parking, building navigation, accessibility, cafeterias, or physical amenities.
{domain_text}

CRITICAL INSTRUCTION FOR MISSING INFORMATION:
- If the best-fit domain label has required slots and the request is missing any of them, you MUST return "Clarification Needed" instead.

ALLOWED LABEL NAMES — your predicted_label MUST be exactly one of these, character-for-character:
{allowed_labels_text}

Output your response ONLY as a valid JSON object with these exact keys:
{{
    "predicted_label": "MUST be one of the allowed label names above — do not invent new names",
    "confidence_level": "High, Medium, or Low",
    "missing_slots": ["List missing required slots, or empty [] if none"],
    "short_reason": "Brief explanation of why you chose this label."
}}"""

        return system_prompt

    def check_gate(self, user_prompt):
        """
        Step A — Gate.
        Decides whether the prompt has enough information to route confidently.
        Returns True (enough info) or False (not enough → Clarification Needed).
        Defaults to True on failure so no prompt is silently dropped.
        """
        definitions_text = ""
        for domain_name, domain_data in self.taxonomy['domains'].items():
            definitions_text += f"\n### {domain_name.upper().replace('_', ' ')} ###\n"
            for label in domain_data['labels']:
                definitions_text += f"- {label['name']}: {label['definition']}\n"

        gate_prompt = f"""You are a gating agent for a routing system.
Your job is to decide if the user request contains enough information to confidently match it to exactly one of the labels below.

ROUTING LABELS:
{definitions_text}

Answer "false" (not enough information) when ANY of these conditions are met:
- The request is so vague that no subject or intent can be identified
  (e.g. "I need help", "something is wrong", "can you check this")
- The intent is completely unclear even after reading the full message
- The intent is clear but the request is missing a specific referent
  needed to route it to exactly one label
  (e.g. "Is my prescription ready?" — intent is clear but which prescription,
  which patient, which pharmacy is unknown;
  "What's on the test?" — intent is clear but which exam, which course is unknown;
  "Where are you located?" — intent is clear but which clinic or location is unknown)
- The request matches two or more labels equally well and no signal in the
  message clearly distinguishes which one applies
  (e.g. a message that could be either a policy question or an escalation,
  or either a concept question or a debugging question, with no disambiguating detail)

Answer "true" (enough information) when:
- The request clearly matches one label based on its definition, even if minor details are missing
- The subject and intent are both clear enough to commit to exactly one label without guessing
- The request is asking for medical diagnosis, clinical interpretation, or treatment advice —
  even if brief, these are clearly identifiable as clinical questions and must pass through
  to be refused appropriately (e.g. "Do I have diabetes?", "Is my rash serious?",
  "What does my MRI result mean?", "Should I take this medication?")

Respond ONLY as a valid JSON object:
{{
    "has_enough_info": true or false,
    "reason": "One short sentence explaining why."
}}

USER REQUEST: "{user_prompt}"
"""

        payload = {
            "model": self.model_name,
            "prompt": gate_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "seed": 42
            }
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = json.loads(response.json().get("response", "{}"))
            return result.get("has_enough_info", True)
        except Exception as e:
            tqdm.write(f"Gate error for prompt '{user_prompt[:40]}...' -> {e}")
            return True  # fail open — let it through to the router

    def route_request(self, user_prompt, ablation=None):
        """
        Four-step routing:
          Safety Override  — pre-gate keyword check for physical emergencies / psychiatric crises
          Academic Override — pre-gate keyword check for exam cheating / academic integrity
          Step A           — Gate: check if enough information is present
          Step B           — Route: pick a label, validate against taxonomy

        ablation parameter (used by ablation experiments only; default None = full system):
          None                     — full system, all components active
          "no_safety"              — skip safety keyword override
          "no_academic"            — skip academic keyword override
          "no_gate"                — skip clarification gate
          "no_hallucination_guard" — skip hallucination guard (use raw LLM label as-is)
          "no_all_gates"           — skip safety override, academic override, and clarification gate (keep hallucination guard)
          "no_all"                 — skip all four components (baseline LLM only)
        """

        # ── PHYSICAL SAFETY OVERRIDE (pre-gate) ───────────────────────────────
        if ablation not in ("no_safety", "no_all_gates", "no_all"):
            safety_label = self._check_safety_override(user_prompt)
            if safety_label:
                return {
                    "initial_label": safety_label,
                    "predicted_label": safety_label,
                    "confidence_level": "High",
                    "missing_slots": [],
                    "short_reason": "Safety override: message contains an unambiguous physical emergency or psychiatric crisis signal.",
                    "was_corrected": False,
                    "qa_reason": "Pre-gate safety keyword match — bypassed gate and router."
                }

        # ── ACADEMIC SAFETY OVERRIDE (pre-gate) ───────────────────────────────
        if ablation not in ("no_academic", "no_all_gates", "no_all"):
            academic_label = self._check_academic_override(user_prompt)
            if academic_label:
                return {
                    "initial_label": academic_label,
                    "predicted_label": academic_label,
                    "confidence_level": "High",
                    "missing_slots": [],
                    "short_reason": "Academic override: message contains an active exam cheating attempt or academic integrity violation signal.",
                    "was_corrected": False,
                    "qa_reason": "Pre-gate academic keyword match — bypassed gate and router."
                }

        # ── STEP A: Gate ───────────────────────────────────────────────────────
        if ablation not in ("no_gate", "no_all_gates", "no_all"):
            has_enough_info = self.check_gate(user_prompt)
            if not has_enough_info:
                return {
                    "initial_label": "Clarification Needed",
                    "predicted_label": "Clarification Needed",
                    "confidence_level": "High",
                    "missing_slots": [],
                    "short_reason": "Request lacks sufficient information to pass the initial Gate.",
                    "was_corrected": False,
                    "qa_reason": "Blocked by gate."
                }

        # ── STEP B: Route ──────────────────────────────────────────────────────
        system_prompt = self.build_system_prompt()
        full_prompt = f"{system_prompt}\n\nUSER REQUEST:\n\"{user_prompt}\""

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "seed": 42
            }
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            initial_output = json.loads(response.json().get("response", "{}"))
        except Exception as e:
            tqdm.write(f"Router error for prompt '{user_prompt[:40]}...' -> {e}")
            return {
                "initial_label": "Error",
                "predicted_label": "Error",
                "confidence_level": "Low",
                "missing_slots": [],
                "short_reason": f"API Error: {str(e)}",
                "was_corrected": False,
                "qa_reason": "N/A"
            }

        raw_label = initial_output.get("predicted_label", "")
        missing_slots = initial_output.get("missing_slots", [])

        # ── Hallucination guard ────────────────────────────────────────────────
        if ablation not in ("no_hallucination_guard", "no_all"):
            validated_label, was_corrected = self._validate_label(raw_label)
        else:
            validated_label, was_corrected = raw_label, False

        qa_reason = (
            f"Label '{raw_label}' not in taxonomy — corrected to '{validated_label}'."
            if was_corrected else ""
        )

        return {
            "initial_label": raw_label,
            "predicted_label": validated_label,
            "confidence_level": initial_output.get("confidence_level", "Unknown"),
            "missing_slots": missing_slots,
            "short_reason": initial_output.get("short_reason", ""),
            "was_corrected": was_corrected,
            "qa_reason": qa_reason
        }

    def evaluate_benchmark(self, input_csv, output_csv):
        """Runs the router over the full benchmark and saves results."""
        print(f"Loading benchmark from {input_csv}...")
        df = pd.read_csv(input_csv)
        results = []

        print(f"Routing {len(df)} requests...\n")

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing", unit="prompt"):
            prompt_text = row['user_prompt']
            domain = row['domain']

            llm_output = self.route_request(prompt_text)

            results.append({
                "prompt_id": row.get('prompt_id', f"ID-{index}"),
                "domain": domain,
                "user_prompt": prompt_text,
                "gold_outcome": row.get('gold_outcome', ''),
                "final_predicted_label": llm_output.get("predicted_label", ""),
                "initial_label": llm_output.get("initial_label", ""),
                "was_corrected": llm_output.get("was_corrected", False),
                "missing_slots": ", ".join(llm_output.get("missing_slots", [])),
                "confidence_level": llm_output.get("confidence_level", ""),
                "short_reason": llm_output.get("short_reason", ""),
                "qa_reason": llm_output.get("qa_reason", "")
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")

        correct = (results_df['gold_outcome'] == results_df['final_predicted_label']).sum()
        total = len(results_df)
        print(f"Accuracy: {correct}/{total} ({(correct/total)*100:.2f}%)")

        n_corrected = results_df['was_corrected'].sum()
        print(f"Hallucination corrections applied: {n_corrected}/{total}")

        try:
            from sklearn.metrics import classification_report
            print(classification_report(
                results_df['gold_outcome'],
                results_df['final_predicted_label'],
                zero_division=0
            ))
        except ImportError:
            pass


# ── Interactive mode ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    taxonomy_path = os.path.abspath(os.path.join(script_dir, "../Data/taxonomy_phase5.json"))

    if not os.path.exists(taxonomy_path):
        taxonomy_path = "taxonomy_phase5.json"

    router = LLMRouterV2(taxonomy_path=taxonomy_path)

    print("\n" + "=" * 50)
    print("LLM Router V2 — Safety Override + Academic Override + Gate + Route + Hallucination Guard")
    print("Type 'quit' to exit.")
    print("=" * 50 + "\n")

    while True:
        user_input = input("Enter your request: ")
        if user_input.strip().lower() in ['quit', 'exit']:
            break
        if not user_input.strip():
            continue

        result = router.route_request(user_input)

        print("\n--- Prediction ---")
        print(f"Predicted Label: {result['predicted_label']}")
        if result.get('was_corrected'):
            print(f"  ⚠ Corrected from: '{result['initial_label']}'")
        print(f"Confidence:      {result['confidence_level']}")
        print(f"Reasoning:       {result['short_reason']}")

        if result.get('missing_slots'):
            print(f"\n⚠️ ACTION REQUIRED: Missing Information ⚠️")
            print(f"Please provide the following to proceed: {', '.join(result['missing_slots'])}")

        print("-" * 25 + "\n")