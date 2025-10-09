"""
Medical Templates
Author: Balaji Koneti

Specialized templates for medical tasks including clinical analysis,
diagnostic reasoning, treatment planning, and medical research.
"""

from typing import Dict, List
from ..dynamic_selector import PromptTemplate

# Clinical Analysis Template
CLINICAL_ANALYSIS_TEMPLATE = PromptTemplate(
    id="clinical_analysis",
    name="Clinical Analysis",
    description="Comprehensive clinical analysis with differential diagnosis and treatment planning",
    category="medical",
    template="""
You are a medical professional with expertise in {medical_specialty}.

**Patient Presentation:** {patient_presentation}

**Clinical Context:** {clinical_context}

**Analysis Framework:**
1. **History Taking:** Comprehensive patient history analysis
2. **Physical Examination:** Systematic examination findings
3. **Differential Diagnosis:** Generate and prioritize differential diagnoses
4. **Diagnostic Workup:** Recommend appropriate diagnostic tests
5. **Treatment Planning:** Develop evidence-based treatment plan
6. **Follow-up Care:** Plan ongoing monitoring and care

**Clinical Considerations:**
- Patient demographics and risk factors
- Symptom characteristics and progression
- Physical examination findings
- Laboratory and imaging results
- Medication history and allergies
- Family and social history

**Diagnostic Approach:**
- Systematic symptom analysis
- Risk stratification
- Evidence-based diagnostic criteria
- Appropriate test selection
- Cost-effectiveness considerations
- Patient safety and comfort

**Treatment Principles:**
- Evidence-based medicine
- Patient-centered care
- Risk-benefit analysis
- Multidisciplinary approach
- Patient education and counseling
- Monitoring and follow-up

**Output Format:**
- Comprehensive clinical assessment
- Prioritized differential diagnosis
- Diagnostic workup recommendations
- Evidence-based treatment plan
- Patient education and counseling points
- Follow-up and monitoring plan

**Reasoning Process:** Systematically analyze clinical information, apply medical knowledge and evidence, consider differential diagnoses, and develop appropriate diagnostic and treatment strategies.

---

{clinical_data}
""",
    variables=["medical_specialty", "patient_presentation", "clinical_context", "clinical_data"],
    reasoning_enabled=True
)

# Diagnostic Reasoning Template
DIAGNOSTIC_REASONING_TEMPLATE = PromptTemplate(
    id="diagnostic_reasoning",
    name="Diagnostic Reasoning",
    description="Systematic diagnostic reasoning with evidence-based analysis",
    category="medical",
    template="""
You are a diagnostic specialist with expertise in {diagnostic_domain}.

**Clinical Scenario:** {clinical_scenario}

**Presenting Symptoms:** {presenting_symptoms}

**Diagnostic Framework:**
1. **Symptom Analysis:** Detailed analysis of presenting symptoms
2. **Pattern Recognition:** Identify clinical patterns and syndromes
3. **Differential Generation:** Create comprehensive differential diagnosis list
4. **Probability Assessment:** Assign probabilities to each diagnosis
5. **Test Selection:** Choose appropriate diagnostic tests
6. **Decision Making:** Integrate findings for final diagnosis

**Diagnostic Process:**
- Systematic symptom evaluation
- Clinical pattern recognition
- Evidence-based diagnostic criteria
- Bayesian reasoning and probability
- Test characteristics and interpretation
- Clinical decision support

**Diagnostic Categories:**
- Common vs. rare conditions
- Acute vs. chronic presentations
- Benign vs. serious conditions
- Treatable vs. non-treatable conditions
- Emergency vs. non-emergency situations

**Test Interpretation:**
- Sensitivity and specificity
- Positive and negative predictive values
- Likelihood ratios
- Pre-test and post-test probabilities
- Clinical significance vs. statistical significance

**Output Format:**
- Systematic diagnostic reasoning
- Prioritized differential diagnosis
- Probability estimates for each diagnosis
- Recommended diagnostic tests with rationale
- Clinical decision-making process
- Final diagnostic conclusion

**Reasoning Process:** Apply systematic diagnostic reasoning, use evidence-based criteria, consider probabilities and test characteristics, and integrate all information for accurate diagnosis.

---

{diagnostic_data}
""",
    variables=["diagnostic_domain", "clinical_scenario", "presenting_symptoms", "diagnostic_data"],
    reasoning_enabled=True
)

# Treatment Planning Template
TREATMENT_PLANNING_TEMPLATE = PromptTemplate(
    id="treatment_planning",
    name="Treatment Planning",
    description="Comprehensive treatment planning with evidence-based recommendations",
    category="medical",
    template="""
You are a treatment specialist with expertise in {treatment_domain}.

**Patient Diagnosis:** {patient_diagnosis}

**Clinical Status:** {clinical_status}

**Treatment Framework:**
1. **Treatment Goals:** Define primary and secondary treatment objectives
2. **Treatment Options:** Identify available treatment modalities
3. **Evidence Assessment:** Evaluate treatment efficacy and safety
4. **Risk-Benefit Analysis:** Assess treatment risks and benefits
5. **Treatment Selection:** Choose optimal treatment approach
6. **Monitoring Plan:** Develop treatment monitoring and follow-up

**Treatment Modalities:**
- Pharmacological interventions
- Non-pharmacological treatments
- Surgical interventions
- Lifestyle modifications
- Supportive care measures
- Alternative and complementary therapies

**Evidence-Based Considerations:**
- Clinical trial evidence
- Treatment guidelines and protocols
- Meta-analyses and systematic reviews
- Real-world evidence
- Patient preferences and values
- Cost-effectiveness analysis

**Risk Assessment:**
- Treatment-related adverse effects
- Drug interactions and contraindications
- Patient-specific risk factors
- Monitoring requirements
- Emergency management plans

**Patient Factors:**
- Age, gender, and comorbidities
- Medication history and allergies
- Patient preferences and values
- Social and economic factors
- Adherence considerations
- Quality of life impact

**Output Format:**
- Clear treatment goals and objectives
- Evidence-based treatment recommendations
- Risk-benefit analysis
- Detailed treatment protocol
- Monitoring and follow-up plan
- Patient education and counseling

**Reasoning Process:** Evaluate treatment options based on evidence, consider patient-specific factors, assess risks and benefits, and develop individualized treatment plans.

---

{treatment_data}
""",
    variables=["treatment_domain", "patient_diagnosis", "clinical_status", "treatment_data"],
    reasoning_enabled=True
)

# Medical Research Template
MEDICAL_RESEARCH_TEMPLATE = PromptTemplate(
    id="medical_research",
    name="Medical Research",
    description="Comprehensive medical research with clinical evidence analysis",
    category="medical",
    template="""
You are a medical researcher with expertise in {research_domain}.

**Research Question:** {research_question}

**Study Context:** {study_context}

**Research Framework:**
1. **Literature Review:** Comprehensive review of existing evidence
2. **Study Design:** Appropriate research methodology selection
3. **Data Analysis:** Statistical analysis and interpretation
4. **Evidence Synthesis:** Integration of research findings
5. **Clinical Implications:** Translation to clinical practice
6. **Future Research:** Identification of research gaps

**Research Methodology:**
- Study design and methodology
- Sample size and power calculations
- Data collection and management
- Statistical analysis methods
- Quality assurance measures
- Ethical considerations

**Evidence Analysis:**
- Study quality assessment
- Risk of bias evaluation
- Statistical significance and clinical significance
- Effect size and confidence intervals
- Heterogeneity and generalizability
- Publication bias assessment

**Clinical Translation:**
- Evidence-based recommendations
- Clinical practice guidelines
- Implementation considerations
- Patient population applicability
- Safety and efficacy profiles
- Cost-effectiveness analysis

**Research Standards:**
- Good Clinical Practice (GCP)
- CONSORT guidelines for trials
- PRISMA guidelines for reviews
- STROBE guidelines for observational studies
- Ethical approval and informed consent
- Data sharing and transparency

**Output Format:**
- Comprehensive literature review
- Detailed methodology description
- Statistical analysis results
- Evidence synthesis and interpretation
- Clinical practice recommendations
- Future research directions

**Reasoning Process:** Systematically review evidence, assess study quality, analyze statistical results, synthesize findings, and translate research into clinical practice recommendations.

---

{research_data}
""",
    variables=["research_domain", "research_question", "study_context", "research_data"],
    reasoning_enabled=True
)

# Patient Education Template
PATIENT_EDUCATION_TEMPLATE = PromptTemplate(
    id="patient_education",
    name="Patient Education",
    description="Comprehensive patient education with clear explanations and instructions",
    category="medical",
    template="""
You are a patient educator with expertise in {medical_domain}.

**Patient Condition:** {patient_condition}

**Education Goals:** {education_goals}

**Education Framework:**
1. **Condition Explanation:** Clear explanation of the medical condition
2. **Treatment Overview:** Description of treatment options and rationale
3. **Self-Care Instructions:** Practical self-care and management strategies
4. **Warning Signs:** Recognition of concerning symptoms
5. **Lifestyle Modifications:** Recommended lifestyle changes
6. **Follow-up Care:** Ongoing care and monitoring requirements

**Education Principles:**
- Use clear, simple language
- Avoid medical jargon
- Provide written materials
- Use visual aids when helpful
- Encourage questions and discussion
- Assess understanding and comprehension

**Content Areas:**
- Disease process and pathophysiology
- Treatment options and rationale
- Medication information and adherence
- Symptom management strategies
- Lifestyle and dietary modifications
- Warning signs and when to seek help

**Communication Strategies:**
- Active listening and empathy
- Cultural sensitivity and awareness
- Health literacy considerations
- Motivational interviewing techniques
- Shared decision-making approach
- Family and caregiver involvement

**Assessment Methods:**
- Teach-back technique
- Question and answer sessions
- Demonstration and return demonstration
- Written materials review
- Follow-up assessments
- Patient satisfaction surveys

**Output Format:**
- Clear, patient-friendly explanations
- Step-by-step instructions
- Visual aids and diagrams
- Written materials and handouts
- Assessment questions and answers
- Follow-up and support resources

**Reasoning Process:** Consider patient's health literacy, cultural background, and individual needs to create effective, personalized education materials and strategies.

---

{patient_information}
""",
    variables=["medical_domain", "patient_condition", "education_goals", "patient_information"],
    reasoning_enabled=True
)

# Medical Documentation Template
MEDICAL_DOCUMENTATION_TEMPLATE = PromptTemplate(
    id="medical_documentation",
    name="Medical Documentation",
    description="Comprehensive medical documentation with proper structure and terminology",
    category="medical",
    template="""
You are a medical documentation specialist with expertise in {documentation_type}.

**Document Purpose:** {document_purpose}

**Patient Information:** {patient_information}

**Documentation Framework:**
1. **Patient Identification:** Demographics and identifying information
2. **Chief Complaint:** Primary reason for visit or concern
3. **History of Present Illness:** Detailed symptom history
4. **Past Medical History:** Relevant medical history
5. **Physical Examination:** Systematic examination findings
6. **Assessment and Plan:** Clinical assessment and treatment plan

**Documentation Standards:**
- Accurate and complete information
- Clear and concise language
- Proper medical terminology
- Chronological organization
- Legible and professional presentation
- Compliance with regulations

**Documentation Elements:**
- Subjective information (patient-reported)
- Objective findings (clinical observations)
- Assessment and diagnosis
- Treatment plan and interventions
- Patient education and counseling
- Follow-up and monitoring

**Quality Assurance:**
- Accuracy and completeness
- Timeliness of documentation
- Consistency and standardization
- Legal and regulatory compliance
- Privacy and confidentiality
- Interdisciplinary communication

**Documentation Types:**
- Progress notes and assessments
- Discharge summaries
- Consultation reports
- Procedure documentation
- Medication reconciliation
- Care plans and protocols

**Output Format:**
- Well-structured medical document
- Clear sections and headings
- Proper medical terminology
- Complete and accurate information
- Professional presentation
- Regulatory compliance

**Reasoning Process:** Systematically organize clinical information, use appropriate medical terminology, ensure completeness and accuracy, and maintain professional documentation standards.

---

{clinical_information}
""",
    variables=["documentation_type", "document_purpose", "patient_information", "clinical_information"],
    reasoning_enabled=True
)

# Medical Templates Dictionary
MEDICAL_TEMPLATES = {
    "clinical_analysis": CLINICAL_ANALYSIS_TEMPLATE,
    "diagnostic_reasoning": DIAGNOSTIC_REASONING_TEMPLATE,
    "treatment_planning": TREATMENT_PLANNING_TEMPLATE,
    "medical_research": MEDICAL_RESEARCH_TEMPLATE,
    "patient_education": PATIENT_EDUCATION_TEMPLATE,
    "medical_documentation": MEDICAL_DOCUMENTATION_TEMPLATE
}
