import os
import asyncio
import subprocess
from pathlib import Path
import chainlit as cl
from capability_extractor import create_retriever_from_file, extract_capability
from open_deep_research_buildorbuy_py import run_graph_and_show_report
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Chat initialization
@cl.on_chat_start
async def start():
    cl.user_session.set("file_received", False)
    cl.user_session.set("capability_extracted", False)
    
    welcome_message = """
    👋 Welcome to the Build or Buy Assistant!

    This tool helps you make informed decisions about whether to build a system or a capability in-house or buy an existing solution.

    How it works:
    1. Upload a document containing your user stories or technical specifications
    2. The assistant will analyze the document to understand the core capability
    3. You'll receive a detailed build or buy analysis including:
       - Market research on existing solutions
       - Cost comparisons
       - Implementation considerations
       - Recommendations based on your specific needs

    Let's get started! Please upload your document (PDF, TXT, or MD format) describing the user stories or technical specifications.
    """
    
    await cl.Message(content=welcome_message).send()
    
    try:
        files = await cl.AskFileMessage(
            content="📄 Upload your requirements document (PDF, TXT, or MD format)",
            accept=["text/plain", "application/pdf", ".md"],
            max_size_mb=20,
            timeout=180,
        ).send()

        if files:
            file = files[0]
            try:
                retriever = create_retriever_from_file(file.path)
                result = extract_capability(retriever)
                
                await cl.Message(
                    content=f"""
## Analysis Results

### System Description
{result['system_description']}

### Capability
**{result['capability']}**
                    """
                ).send()

                cl.user_session.set("capability", result['capability'])
                cl.user_session.set("file_received", True)
                await asyncio.sleep(1)
                
                await cl.Message(
                    content=f"Would you like to run a build or buy analysis report for this capability: **{result['capability']}**?",
                    actions=[
                        cl.Action(name="run_analysis", value="yes", label="✅ Run Analysis", payload={"action": "run"}),
                        cl.Action(name="cancel_analysis", value="no", label="❌ Cancel", payload={"action": "cancel"})
                    ]
                ).send()
            except Exception as e:
                await cl.Message(f"Error processing file: {str(e)}").send()
    except Exception as e:
        await cl.Message(f"Error receiving file: {str(e)}").send()

# Action handlers
@cl.action_callback("run_analysis")
async def on_run_analysis(action):
    capability = cl.user_session.get("capability")
    await cl.Message("🔍 Starting the build or buy analysis report...").send()
    
    try:
        result = await run_graph_and_show_report(capability=capability, auto_approve_plan=True)
        
        if isinstance(result, dict) and 'final_report' in result:
            report_content = result['final_report']
            await cl.Message(content=f"# Build vs Buy Analysis Report: {capability}\n\n{report_content}").send()
            
            try:
                pdf_file = await convert_to_slides_and_get_file(report_content, capability)
                file_element = cl.File(name=pdf_file.name, path=str(pdf_file), display="inline")
                await cl.Message(content="📊 Download the slide deck version of the report", elements=[file_element]).send()
                pdf_file.unlink(missing_ok=True)
            except Exception as e:
                await cl.Message(content=f"⚠️ Could not generate PDF slides: {str(e)}").send()
        else:
            await cl.Message(content="❌ Error: Could not generate final report.").send()
    except Exception as e:
        await cl.Message(f"❌ Error during analysis: {str(e)}").send()

@cl.action_callback("cancel_analysis")
async def on_cancel_analysis(action):
    await cl.Message("👋 Thank you for using the app!").send()

# Slide generation utilities
async def convert_to_slides_and_get_file(report_content: str, capability: str) -> Path:
    """Convert report to slides and return the PDF file path."""
    slides_dir = Path("slides")
    slides_dir.mkdir(exist_ok=True)
    
    filename = f"build_vs_buy_{capability.lower().replace(' ', '_')}"
    md_file = slides_dir / f"{filename}.md"
    pdf_file = slides_dir / f"{filename}.pdf"

    # Initialize ChatOpenAI
    summarizer = ChatOpenAI(
        model="gpt-4-turbo-preview", 
        temperature=0
    )
    
    # Create summary prompt
    summary_prompt = ChatPromptTemplate.from_template("""
    Summarize this build vs buy analysis report into key points for a presentation. 
    For each section, provide 3-4 bullet points of the most important insights.
    
    Format the output as:
    
    Executive Summary:
    - [3 key takeaways from entire report]
    
    Buy Options:
    - [key points about buy options]
    
    Build Options:
    - [key points about build options]
    
    Recommendation:
    - [final recommendation and rationale]
    
    Report content:
    {report_content}
    """)
    
    # Generate summary
    chain = summary_prompt | summarizer | StrOutputParser()
    summary = await chain.ainvoke({"report_content": report_content})
    
    # Create slides content
    slides = [
        f"""---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

<!-- _class: lead -->
# Build vs Buy Analysis
## {capability}
""",
        """---
# Executive Summary
"""
    ]
    
    # Process summary into slides
    current_section = None
    current_points = []
    
    for line in summary.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.endswith(':'):  # This is a section header
            # Add previous section if it exists
            if current_section and current_points:
                slides.append("\n".join([f"# {current_section}", *current_points, "---"]))
                current_points = []
            current_section = line[:-1]  # Remove the colon
        elif line.startswith('- '):
            current_points.append(line)
            
            # Create new slide if we have 4 points
            if len(current_points) == 4:
                slides.append("\n".join([f"# {current_section}", *current_points, "---"]))
                current_points = []
                if current_section:  # Add continued header for next slide
                    current_section = f"{current_section} (continued)"
    
    # Add remaining points
    if current_section and current_points:
        slides.append("\n".join([f"# {current_section}", *current_points]))
    
    # Add a final recommendation slide
    slides.append("""---
# Next Steps

- Review the detailed analysis in the full report
- Engage with stakeholders for feedback
- Create implementation timeline
- Define success metrics
""")
    
    # Write to markdown file
    md_content = "\n\n".join(slides)
    md_file.write_text(md_content)
    
    try:
        subprocess.run(["marp", str(md_file), "--pdf", "-o", str(pdf_file)], check=True)
        return pdf_file
    except Exception as e:
        print(f"Error converting to PDF: {e}")
        raise
    finally:
        md_file.unlink(missing_ok=True)