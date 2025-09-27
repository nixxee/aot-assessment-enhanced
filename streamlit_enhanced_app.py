# streamlit_app.py - Enhanced Multi-AI AOT Assessment with Batch Processing

import streamlit as st
import pandas as pd
import openai
from google import generativeai as genai
import anthropic
import time
import traceback
import re
from typing import Dict, List
from io import StringIO
import json

# Page config
st.set_page_config(
    page_title="Multi-AI AOT Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .batch-stats {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .mode-selector {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# AOT Framework
AOT_SCORING_GUIDE = """
Actively Open-Minded Thinking (AOT) Assessment Scale (1-7):

1-2 (Very Low AOT): 
- Rigid thinking, dismisses opposing views
- Overconfident in own judgment
- Rarely seeks other perspectives
- Denies making mistakes or learning from them

3-4 (Low AOT):
- Some consideration of alternatives but limited
- Moderate confidence, occasional self-doubt
- Sometimes seeks input but may dismiss it
- Acknowledges some mistakes but defensively

5 (Moderate AOT):
- Balanced consideration of multiple perspectives
- Appropriate confidence levels
- Regularly seeks input from others
- Learns from mistakes without being overly self-critical

6-7 (High AOT):
- Actively seeks disconfirming evidence
- Intellectual humility, comfortable with uncertainty
- Values diverse perspectives highly
- Views mistakes as learning opportunities

Provide your score with detailed reasoning.
"""

# API Functions
@st.cache_data(ttl=300)
def test_apis(api_keys):
    working_apis = {}
    status_messages = {}
    
    # Test OpenAI
    if api_keys.get('openai'):
        try:
            client = openai.OpenAI(api_key=api_keys['openai'])
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            working_apis['gpt'] = True
            status_messages['gpt'] = "✅ Working"
        except Exception as e:
            working_apis['gpt'] = False
            status_messages['gpt'] = f"❌ {str(e)[:50]}..."
    else:
        working_apis['gpt'] = False
        status_messages['gpt'] = "⚪ No API key provided"
    
    # Test Claude
    if api_keys.get('anthropic'):
        claude_models = ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-haiku-20240307"]
        working_apis['claude'] = False
        try:
            anthropic_client = anthropic.Anthropic(api_key=api_keys['anthropic'])
            for model in claude_models:
                try:
                    response = anthropic_client.messages.create(
                        model=model, max_tokens=5,
                        messages=[{"role": "user", "content": "Test"}]
                    )
                    working_apis['claude'] = model
                    status_messages['claude'] = f"✅ Working with {model}"
                    break
                except:
                    continue
            if not working_apis['claude']:
                status_messages['claude'] = "❌ No accessible models"
        except Exception as e:
            working_apis['claude'] = False
            status_messages['claude'] = f"❌ {str(e)[:50]}..."
    else:
        working_apis['claude'] = False
        status_messages['claude'] = "⚪ No API key provided"
    
    # Test Gemini
    if api_keys.get('google'):
        gemini_models = ['gemini-2.0-flash-exp', 'gemini-1.5-pro']
        working_apis['gemini'] = False
        try:
            genai.configure(api_key=api_keys['google'])
            for model_name in gemini_models:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content("Test")
                    if response.text:
                        working_apis['gemini'] = model_name
                        status_messages['gemini'] = f"✅ Working with {model_name}"
                        break
                except:
                    continue
            if not working_apis['gemini']:
                status_messages['gemini'] = "❌ No accessible models"
        except Exception as e:
            working_apis['gemini'] = False
            status_messages['gemini'] = f"❌ {str(e)[:50]}..."
    else:
        working_apis['gemini'] = False
        status_messages['gemini'] = "⚪ No API key provided"
    
    return working_apis, status_messages

def call_gpt_api(prompt: str, api_key: str, model: str = "gpt-3.5-turbo") -> str:
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT API Error: {str(e)}"

def call_claude_api(prompt: str, api_key: str, model_name: str) -> str:
    try:
        anthropic_client = anthropic.Anthropic(api_key=api_key)
        response = anthropic_client.messages.create(
            model=model_name, max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Claude API Error: {str(e)}"

def call_gemini_api(prompt: str, api_key: str, model_name: str) -> str:
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API Error: {str(e)}"

# Score extraction
def extract_aot_score(text: str) -> float:
    if not text or "Error" in text:
        return None
    patterns = [
        r'AOT Score:\s*(\d+(?:\.\d+)?)',
        r'score of (\d+(?:\.\d+)?)',
        r'score:\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)/7',
        r'(\d+(?:\.\d+)?)\s*out of 7'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            if 1 <= score <= 7:
                return score
    return None

# Assessment functions
def get_single_assessment(ai_name: str, interview_text: str, working_apis: dict, api_keys: dict, model_settings: dict = None) -> str:
    prompt = f"""You are an expert psychologist specializing in Actively Open-Minded Thinking (AOT) assessment.

{AOT_SCORING_GUIDE}

Please analyze the following interview responses for AOT indicators:

{interview_text}

Provide:
1. Your AOT score (1-7) - be very clear about this number
2. Key evidence from the text
3. Reasoning for your score
4. Specific quotes that support your assessment

Be thorough and specific in your analysis. Make sure to clearly state your numerical AOT score."""
    
    gpt_model = model_settings.get('gpt_model', 'gpt-3.5-turbo') if model_settings else 'gpt-3.5-turbo'
    
    if ai_name == 'gpt' and working_apis.get('gpt'):
        return call_gpt_api(prompt, api_keys['openai'], gpt_model)
    elif ai_name == 'claude' and working_apis.get('claude'):
        return call_claude_api(prompt, api_keys['anthropic'], working_apis['claude'])
    elif ai_name == 'gemini' and working_apis.get('gemini'):
        return call_gemini_api(prompt, api_keys['google'], working_apis['gemini'])
    else:
        return f"{ai_name.upper()} not available"

def get_refined_assessment(ai_name: str, interview_text: str, other_assessments: dict, working_apis: dict, api_keys: dict, model_settings: dict = None) -> str:
    all_assessments = "\n".join([
        f"{ai.upper()} ASSESSMENT:\n{assessment}\n"
        for ai, assessment in other_assessments.items() if ai != ai_name
    ])
    
    prompt = f"""Review these AOT assessments from other AI systems and provide your refined analysis:

INTERVIEW TEXT:
{interview_text}

OTHER AI ASSESSMENTS:
{all_assessments}

Instructions:
1. Consider the perspectives of the other AIs
2. Identify points of agreement and disagreement  
3. Provide your final AOT score (1-7) with reasoning
4. Explain how other assessments influenced your thinking
5. Be specific about evidence from the text

{AOT_SCORING_GUIDE}

Make sure to clearly state your refined numerical AOT score."""
    
    gpt_model = model_settings.get('gpt_model', 'gpt-3.5-turbo') if model_settings else 'gpt-3.5-turbo'
    
    if ai_name == 'gpt' and working_apis.get('gpt'):
        return call_gpt_api(prompt, api_keys['openai'], gpt_model)
    elif ai_name == 'claude' and working_apis.get('claude'):
        return call_claude_api(prompt, api_keys['anthropic'], working_apis['claude'])
    elif ai_name == 'gemini' and working_apis.get('gemini'):
        return call_gemini_api(prompt, api_keys['google'], working_apis['gemini'])
    else:
        return f"{ai_name.upper()} not available"

def get_consensus_assessment(interview_text: str, refined_assessments: dict, working_apis: dict, api_keys: dict, model_settings: dict = None) -> str:
    all_refined = "\n".join([
        f"{ai.upper()} REFINED ASSESSMENT:\n{assessment}\n"
        for ai, assessment in refined_assessments.items()
    ])
    
    prompt = f"""Based on all refined assessments below, provide a final consensus AOT score:

INTERVIEW TEXT:
{interview_text}

ALL REFINED ASSESSMENTS:
{all_refined}

Instructions:
1. Identify the final consensus AOT score (1-7)
2. Explain the reasoning behind this consensus
3. Note any remaining disagreements
4. Provide strongest evidence from the interview
5. Give confidence level (1-10)

This should be the definitive AOT assessment. Make sure to clearly state your final consensus numerical AOT score."""
    
    available_ais = [ai for ai in ['gpt', 'claude', 'gemini'] if working_apis.get(ai)]
    if not available_ais:
        return "No AIs available for consensus"
    
    consensus_ai = available_ais[0]
    gpt_model = model_settings.get('gpt_model', 'gpt-3.5-turbo') if model_settings else 'gpt-3.5-turbo'
    
    if consensus_ai == 'gpt':
        return call_gpt_api(prompt, api_keys['openai'], gpt_model)
    elif consensus_ai == 'claude':
        return call_claude_api(prompt, api_keys['anthropic'], working_apis['claude'])
    elif consensus_ai == 'gemini':
        return call_gemini_api(prompt, api_keys['google'], working_apis['gemini'])

def process_single_row_full_consensus(interview_text: str, working_apis: dict, api_keys: dict, model_settings: dict = None):
    results = {
        "gpt_initial_assessment": "", "claude_initial_assessment": "", "gemini_initial_assessment": "",
        "gpt_initial_score": None, "claude_initial_score": None, "gemini_initial_score": None,
        "gpt_refined_assessment": "", "claude_refined_assessment": "", "gemini_refined_assessment": "",
        "gpt_refined_score": None, "claude_refined_score": None, "gemini_refined_score": None,
        "consensus_assessment": "", "consensus_score": None, "available_ais": []
    }
    
    for ai_name in ['gpt', 'claude', 'gemini']:
        if working_apis.get(ai_name):
            results["available_ais"].append(ai_name)
    
    if not results["available_ais"]:
        return results
    
    # Phase 1: Initial assessments
    initial_assessments = {}
    for ai_name in results["available_ais"]:
        assessment = get_single_assessment(ai_name, interview_text, working_apis, api_keys, model_settings)
        results[f"{ai_name}_initial_assessment"] = assessment
        results[f"{ai_name}_initial_score"] = extract_aot_score(assessment)
        initial_assessments[ai_name] = assessment
    
    # Phase 2: Refined assessments
    if len(results["available_ais"]) > 1:
        refined_assessments = {}
        for ai_name in results["available_ais"]:
            assessment = get_refined_assessment(ai_name, interview_text, initial_assessments, working_apis, api_keys, model_settings)
            results[f"{ai_name}_refined_assessment"] = assessment
            results[f"{ai_name}_refined_score"] = extract_aot_score(assessment)
            refined_assessments[ai_name] = assessment
        
        # Phase 3: Consensus
        consensus = get_consensus_assessment(interview_text, refined_assessments, working_apis, api_keys, model_settings)
        results["consensus_assessment"] = consensus
        results["consensus_score"] = extract_aot_score(consensus)
    else:
        single_ai = results["available_ais"][0]
        results["consensus_assessment"] = f"Single AI assessment: {initial_assessments[single_ai]}"
        results["consensus_score"] = results[f"{single_ai}_initial_score"]
    
    return results

def process_batch(df: pd.DataFrame, question_answer_columns: list, working_apis: dict, api_keys: dict, model_settings: dict):
    # Add columns
    for ai in ['gpt', 'claude', 'gemini']:
        if working_apis.get(ai):
            df[f'{ai}_initial_assessment'] = ''
            df[f'{ai}_initial_score'] = None
            df[f'{ai}_refined_assessment'] = ''
            df[f'{ai}_refined_score'] = None
    
    df['consensus_assessment'] = ''
    df['consensus_score'] = None
    df['processing_status'] = 'pending'
    df['gpt_model_used'] = model_settings.get('gpt_model', 'gpt-3.5-turbo')
    
    progress_bar = st.progress(0)
    status_container = st.empty()
    
    for index, row in df.iterrows():
        try:
            interview_parts = []
            for i, (q_col, a_col) in enumerate(question_answer_columns, 1):
                question = str(row[q_col]) if pd.notna(row[q_col]) else f"Question {i}:"
                answer = str(row[a_col]) if pd.notna(row[a_col]) else ""
                if answer.strip():
                    interview_parts.append(f"Question {i}: {question}\n\nAnswer: {answer}")
            
            if not interview_parts:
                df.at[index, 'processing_status'] = 'skipped_empty'
                continue
            
            interview_text = "\n\n".join(interview_parts)
            gpt_model = model_settings.get('gpt_model', 'gpt-3.5-turbo')
            status_container.text(f"Processing row {index + 1}/{len(df)}: Full consensus with {gpt_model}...")
            
            results = process_single_row_full_consensus(interview_text, working_apis, api_keys, model_settings)
            
            for ai in results["available_ais"]:
                df.at[index, f'{ai}_initial_assessment'] = results[f'{ai}_initial_assessment']
                df.at[index, f'{ai}_initial_score'] = results[f'{ai}_initial_score']
                df.at[index, f'{ai}_refined_assessment'] = results[f'{ai}_refined_assessment']
                df.at[index, f'{ai}_refined_score'] = results[f'{ai}_refined_score']
            
            df.at[index, 'consensus_assessment'] = results['consensus_assessment']
            df.at[index, 'consensus_score'] = results['consensus_score']
            df.at[index, 'processing_status'] = 'completed'
            
        except Exception as e:
            df.at[index, 'processing_status'] = f'error: {str(e)[:50]}'
        
        progress_bar.progress((index + 1) / len(df))
    
    status_container.text("✅ Full consensus batch processing complete!")
    return df

# Main app
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Multi-AI AOT Assessment</h1>
        <p>Actively Open-Minded Thinking Analysis with GPT, Claude & Gemini</p>
        <p><small>Now with Batch Processing Support!</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
    mode = st.radio("🔧 **Select Assessment Mode:**", ["📝 Single Assessment", "📊 Batch Processing"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    with st.sidebar.expander("📝 API Keys", expanded=True):
        openai_key = st.text_input("OpenAI API Key", type="password")
        anthropic_key = st.text_input("Anthropic API Key", type="password")
        google_key = st.text_input("Google AI API Key", type="password")
    
    with st.sidebar.expander("🤖 Model Settings", expanded=True):
        gpt_model = st.selectbox(
            "GPT Model:",
            options=["gpt-4", "gpt-3.5-turbo"],
            index=1,
            help="GPT-4: Higher quality, ~20x more expensive. GPT-3.5-Turbo: Good quality, much cheaper"
        )
        
        if gpt_model == "gpt-4":
            st.info("💰 GPT-4: Premium quality, higher cost (~$0.30 per assessment)")
        else:
            st.success("💰 GPT-3.5-Turbo: Good quality, low cost (~$0.01 per assessment)")
    
    api_keys = {'openai': openai_key, 'anthropic': anthropic_key, 'google': google_key}
    model_settings = {'gpt_model': gpt_model}
    
    # Test APIs
    if st.sidebar.button("🧪 Test API Connections", type="secondary"):
        if any(api_keys.values()):
            with st.spinner("Testing API connections..."):
                working_apis, status_messages = test_apis(api_keys)
                st.sidebar.markdown("**API Status:**")
                for ai, status in status_messages.items():
                    st.sidebar.markdown(f"**{ai.upper()}:** {status}")
                st.session_state.working_apis = working_apis
                st.session_state.api_keys = api_keys
        else:
            st.sidebar.error("Please provide at least one API key")
    
    # Main interface
    if mode == "📝 Single Assessment":
        single_assessment_interface(api_keys, model_settings)
    else:
        batch_processing_interface(api_keys, model_settings)

def single_assessment_interface(api_keys, model_settings):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Interview Input")
        interview_text = st.text_area("Interview Text", height=300, placeholder="Paste your interview...")
        
        if st.button("🚀 Start AOT Assessment", type="primary", use_container_width=True):
            if not interview_text.strip():
                st.error("Please enter interview text.")
            elif not any(api_keys.values()):
                st.error("Please configure API keys.")
            else:
                if 'working_apis' not in st.session_state:
                    with st.spinner("Testing APIs..."):
                        working_apis, _ = test_apis(api_keys)
                        st.session_state.working_apis = working_apis
                
                working_apis = st.session_state.working_apis
                if not any(working_apis.values()):
                    st.error("No working APIs found.")
                else:
                    results = process_single_row_full_consensus(interview_text, working_apis, api_keys, model_settings)
                    st.session_state.single_results = results
    
    with col2:
        st.subheader("📊 Assessment Results")
        if 'single_results' in st.session_state:
            results = st.session_state.single_results
            for ai in results["available_ais"]:
                icon = "🤖" if ai == 'gpt' else "🧠" if ai == 'claude' else "💎"
                score = results[f'{ai}_initial_score']
                score_display = f" (Score: {score})" if score else ""
                with st.expander(f"{icon} {ai.upper()}{score_display}", expanded=True):
                    st.write(results[f'{ai}_initial_assessment'])
            if results['consensus_score']:
                st.success(f"🏆 **Consensus AOT Score: {results['consensus_score']}/7**")
        else:
            st.info("👆 Configure API keys and run assessment.")

def batch_processing_interface(api_keys, model_settings):
    st.subheader("📊 Batch Processing")
    st.info("💡 Upload CSV/Excel with interview responses!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("📁 Upload dataset", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ File uploaded! Found {len(df)} rows.")
                
                with st.expander("👀 Data Preview", expanded=True):
                    st.dataframe(df.head())
                
                st.markdown("### 📝 Select Question-Answer Pairs")
                question_answer_pairs = []
                num_pairs = st.number_input("Number of Q&A pairs:", min_value=1, max_value=5, value=2)
                
                for i in range(num_pairs):
                    col1_inner, col2_inner = st.columns(2)
                    with col1_inner:
                        q_col = st.selectbox(f"Question {i+1}:", [''] + df.columns.tolist(), key=f"q_{i}")
                    with col2_inner:
                        a_col = st.selectbox(f"Answer {i+1}:", [''] + df.columns.tolist(), key=f"a_{i}")
                    if q_col and a_col:
                        question_answer_pairs.append((q_col, a_col))
                
                if question_answer_pairs and len(df) > 0:
                    st.markdown("**Sample combined text:**")
                    sample_parts = []
                    for i, (q_col, a_col) in enumerate(question_answer_pairs, 1):
                        q = str(df[q_col].iloc[0]) if pd.notna(df[q_col].iloc[0]) else f"Q{i}:"
                        a = str(df[a_col].iloc[0]) if pd.notna(df[a_col].iloc[0]) else ""
                        if a.strip():
                            sample_parts.append(f"Q{i}: {q}\nA{i}: {a}")
                    sample_text = "\n\n".join(sample_parts)[:300] + "..."
                    st.text_area("Sample:", sample_text, height=150, disabled=True)
                
                max_rows = st.number_input("Max rows (0=all):", min_value=0, max_value=len(df), value=0)
                if max_rows > 0:
                    df = df.head(max_rows)
                
                # Cost estimate
                gpt_model = model_settings.get('gpt_model', 'gpt-3.5-turbo')
                cost_per_row = 0.45 if gpt_model == 'gpt-4' else 0.05
                estimated_cost = len(df) * 3 * cost_per_row  # 3 AIs estimate
                
                st.warning(f"⚠️ **Full Consensus with {gpt_model}** - Estimated: ${estimated_cost:.2f}")
                
                if st.button("🚀 Process Batch", type="primary", use_container_width=True):
                    if not question_answer_pairs:
                        st.error("Select Q&A pairs!")
                    elif not any(api_keys.values()):
                        st.error("Add API keys!")
                    else:
                        if 'working_apis' not in st.session_state:
                            with st.spinner("Testing APIs..."):
                                working_apis, _ = test_apis(api_keys)
                                st.session_state.working_apis = working_apis
                        
                        working_apis = st.session_state.working_apis
                        if not any(working_apis.values()):
                            st.error("No working APIs.")
                        else:
                            st.markdown(f"### 🔄 Processing with {gpt_model}...")
                            start_time = time.time()
                            processed_df = process_batch(df, question_answer_pairs, working_apis, api_keys, model_settings)
                            st.session_state.processed_df = processed_df
                            st.session_state.processing_complete = True
                            end_time = time.time()
                            st.success(f"✅ Complete! Took {(end_time-start_time)/60:.1f} minutes")
                            st.rerun()
                            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        else:
            st.info("👆 Upload CSV/Excel to start")
    
    with col2:
        st.subheader("📈 Results & Download")
        
        if st.session_state.get('processing_complete') and 'processed_df' in st.session_state:
            df_results = st.session_state.processed_df
            completed = len(df_results[df_results['processing_status'] == 'completed'])
            total = len(df_results)
            gpt_model_used = df_results['gpt_model_used'].iloc[0] if 'gpt_model_used' in df_results.columns else "Unknown"
            
            st.markdown(f"""
            <div class="batch-stats">
                <h4>📊 Processing Statistics</h4>
                <ul>
                    <li>✅ <strong>Completed:</strong> {completed}/{total}</li>
                    <li>📝 <strong>Success Rate:</strong> {completed/total*100:.1f}%</li>
                    <li>🤖 <strong>GPT Model:</strong> {gpt_model_used}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("👀 Sample Results", expanded=True):
                score_cols = [col for col in df_results.columns if col.endswith('_score')]
                preview_cols = ['processing_status'] + score_cols
                available_cols = [col for col in preview_cols if col in df_results.columns]
                st.dataframe(df_results[available_cols].head())
            
            csv_data = df_results.to_csv(index=False)
            st.download_button(
                label="💾 Download Complete Results",
                data=csv_data,
                file_name=f"aot_full_consensus_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.markdown("**📋 Download includes:**")
            result_cols = [col for col in df_results.columns if 'assessment' in col or 'score' in col]
            st.write(f"• Original data + {len(result_cols)} assessment columns")
            st.write("• Initial, Refined, and Consensus from each AI")
            
            if st.button("🗑️ Clear Results", type="secondary"):
                for key in ['processed_df', 'processing_complete']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        else:
            st.info("📊 Process a batch to see results.")

if __name__ == "__main__":
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    main()
    st.markdown("---")
    st.markdown("🧠 **Multi-AI AOT Assessment System** • Built with Streamlit")
