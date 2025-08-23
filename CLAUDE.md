# INSTRUTION
Không được code những gì mới và không nằm trong kế hoạc cuả tôi. Khi có đề xuất code thêm những hàm, phần mới, cần phải hỏi ý kiến tôi trước. Hạn chế Error handling và Comment không cần thiết.

CODE REQUIREMENTS:
Always communicate in Vietnamese

## Constants Over Magic Numbers
- No hard-coded values with named constants
- Keep constants at the top of the file

## Smart Comments
- Don't not comment to much. Comment in English
- Use comments to explain why something is done a certain way.

## DRY (Don't Repeat Yourself)
- Extract repeated code into reusable functions
- Share common logic through proper abstraction
- Maintain single sources of truth

## Encapsulation
- Expose clear interfaces
- Move nested conditionals into well-named functions

## KEY NOTES:
- Strictly adhere to the explicitly given instructions. Do not do anything extra.
- Never make medium to large changes based on your own ideas and initiative. Always ask and suggest to me first. => ONLY make changes IF I accepted.
- Work incrementally. Do not try to complete the entire task in one go.
- Do not add comments in code to make notes to me about the changes you made. That goes in the chat not in the code. Only make comments in code as though you are a developer making changes and leaving notes for non-obvious or temporary changes.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Application Overview

This is a Vietnamese teacher assistant application built with Streamlit for automated grading of student exam submissions. The app processes exam papers and student submissions through AI analysis, then provides intelligent grading and feedback.

### Core Architecture

The application follows a 4-step workflow displayed as numbered steps in the UI:

1. **Exam Analysis (Step 1)**: Upload and analyze exam images using OpenAI vision models, parse into individual questions with knowledge topics (3-5 tags required)
2. **Solution Generation (Step 2/3)**: Create standard solutions with reasoning approaches and grading rubrics for each question using GPT-5-mini
3. **Submission Processing (Step 3/4)**: Upload student submission images, process and segment into question-specific answers 
4. **AI Grading (Step 4/5)**: Grade each submission item by comparing with standard solutions, identify knowledge gaps and calculation errors, generate reports and performance analysis

### Key Components

- `app.py`: Main Streamlit application with 4-step workflow, sidebar navigation with step jumping, password protection via Streamlit secrets
- `database/`: SQLAlchemy models and database manager
  - `models.py`: Database models with datetime_now_seconds() helper (Exam, Question, Submission, SubmissionItem, Grading, QuestionSolution, SubmissionReport, PerformanceAnalysis)
  - `db_manager.py`: Database operations with PostgreSQL support, automatic migrations, LaTeX preview formatting
- `services/`: Core business logic services
  - `exam_analyzer.py`: OpenAI vision models for OCR and question analysis from exam images using base64 encoding
  - `grading_service.py`: Core grading logic using GPT-5-mini for advanced grading, delegated report building
  - `grading/`: Modular grading components
    - `report_builder.py`: Dedicated report generation service
    - `statistics_calculator.py`: Statistical analysis utilities
  - `submission_processor.py`: Process and segment student submissions from images
  - `solution_service.py`: Generate standard solutions with contextual analysis using GPT-5-mini
  - `performance_analyzer.py`: Advanced analysis to group common mistakes and knowledge gaps
- `utils/`: Utility modules and configurations
  - `config.py`: Application configuration with model settings (gpt-4.1-mini for analysis, gpt-5-mini for solutions/grading), API keys, database URL, model pricing
  - `constants.py`: Centralized constants, error messages, and application defaults
  - `llm_logger.py`: LLM API call logging with cost tracking
  - `schemas.py`: JSON schemas for OpenAI API responses
  - `data_models.py`: Data models for internal processing (GradingResult, SolutionResult)
  - `prompts.py`: Comprehensive system prompts for various AI tasks
- `delete_gradings.py`: CLI tool for clearing grading data

The database uses PostgreSQL with foreign key relationships and automatic migrations. Session state management allows jumping between workflow steps and preserves user data.

## Development Commands

### Installation & Setup
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
streamlit run app.py
```

### Environment Configuration
Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=your_postgresql_connection_string
```

## Database Schema

The application uses SQLAlchemy with PostgreSQL and the following key models:
- `Exam`: Contains exam metadata and name (no original text stored after migration)
- `Question`: Individual exam questions with difficulty ratings (0=unset, updated when solution created), knowledge topics (JSON string), order_index, and part_label
- `QuestionSolution`: Standard solutions with reasoning_approach (grading rubric) and final_answer
- `Submission`: Student submissions linked to exams with student names and optional original_text
- `SubmissionItem`: Segmented answers for specific questions with position tracking and answer_text
- `Grading`: AI-generated feedback with knowledge_gaps and calculation_logic_errors (JSON arrays), is_correct boolean, and optional final_score
- `SubmissionReport`: Generated markdown reports for students
- `PerformanceAnalysis`: Grouped analysis with group_name, group_type (knowledge/error), description, and related_questions (JSON array)

Database uses PostgreSQL with automatic migrations (removes original_text from exams table) and foreign key relationships with cascade deletes.

## AI Integration

- **Models**: Multiple OpenAI models configured for different tasks:
  - `gpt-4.1-mini`: Exam analysis, submission processing, grouping analysis, report generation (temperature 0.1)
  - `gpt-5-mini`: Solution generation and advanced grading (temperature 0.1 for grading)
- **API Configuration**: OpenAI client initialized per service with OPENAI_API_KEY environment variable
- **Vision OCR**: Uses OpenAI vision models to extract text from exam and submission images with base64 encoding and proper MIME type detection
- **Question Analysis**: Parses Vietnamese exam content using structured JSON schema, extracts individual questions with knowledge topics (3-5 tags required), order_index, and part_label
- **Solution Generation**: Creates comprehensive solutions using contextual analysis from related questions in same order_index, with reasoning_approach (grading rubric), final_answer, and difficulty assessment (1-10 scale for grade 9 students)
- **Advanced Grading**: Analyzes student answers against standard solutions using GPT-5-mini with structured JSON schema, identifies knowledge gaps, calculation/logic errors, and provides correctness assessment
- **Performance Analysis**: Groups common mistakes and knowledge gaps across submissions with database persistence

## Session State Management

The Streamlit app uses extensive session state to:
- Maintain workflow position (`current_step`: 1, 3, 4, 5)
- Store parsed questions and segmented items
- Track exam/submission IDs for database operations  
- Allow navigation between steps via sidebar with database data selection
- Preserve grading results during CSV downloads
- Handle password authentication (`password_correct`)

## Key Features

- 4-step workflow with sidebar navigation for step jumping (steps 1, 3, 4, 5 mapped to current_step)
- Password protection using Streamlit secrets (ACCESS_KEY)
- Multi-image upload support for both exams and submissions with OpenAI vision processing
- Enhanced LaTeX display with interactive line clicking and rendered preview in expandable sections
- Vietnamese language support throughout UI and AI responses  
- PostgreSQL database persistence with ability to resume work from any step
- AI-powered question analysis with knowledge topic extraction (3-5 tags validation) and difficulty assessment
- Contextual solution generation with grading rubrics using GPT-5-mini
- Advanced grading with knowledge gap analysis and calculation/logic error detection
- Performance analysis grouping common mistakes with database persistence
- Editable data tables for parsed questions and segmented submission items
- Report generation with markdown output and download capabilities
- Session state preservation for grading results during CSV exports
- LLM API call logging with cost tracking and model pricing configuration
- Database migration support and CLI utility tools

## Technical Implementation

### Data Flow
1. **Image Processing**: Multi-image uploads → Temporary files → Base64 encoding → OpenAI Vision API with MIME type detection
2. **Question Parsing**: Vision OCR → JSON schema validation → Database storage with knowledge topics validation (3-5 tags)
3. **Solution Generation**: Contextual question analysis → GPT-5-mini with structured schema → Database storage with difficulty update
4. **Submission Segmentation**: Image processing → Question matching → Editable dataframes → Database storage
5. **Grading**: Solution comparison → GPT-5-mini structured analysis → Session state preservation → Optional database save
6. **Reporting**: Grading data aggregation → Report generation → Markdown output with download

### Error Handling
- Structured JSON schema validation for all AI responses
- Try-catch blocks with fallback responses for API failures
- Data type conversion and validation with pandas DataFrames
- Database transaction safety with session management and automatic rollbacks
- Temporary file cleanup after image processing

### Performance Considerations
- PostgreSQL database with foreign key relationships and cascade deletes
- Session state management for large datasets and step navigation
- Efficient database queries with SQLAlchemy joins and filtering
- Cost-optimized model selection (gpt-4.1-mini for analysis, gpt-5-mini for grading)
- Temperature consistency (0.1) for reproducible results
- Modular service architecture with clear separation of concerns
- LLM call logging for cost tracking and debugging