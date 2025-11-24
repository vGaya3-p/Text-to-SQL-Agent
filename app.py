import os
import re
import time
import logging
from typing import TypedDict, Dict, Any, Optional, Tuple
import sqlglot
from sqlglot import exp
from flask import Flask, request, jsonify, render_template, send_from_directory, abort
from dotenv import load_dotenv


# --- LangSmith imports and setup ---
try:
    import langsmith
    from langchain_core.callbacks.manager import tracing_v2_enabled
    from langchain_core.tracers.langchain import LangChainTracer
    LANGSMITH_AVAILABLE = True
except ImportError as e:
    print(f"LangSmith import error: {e}")
    langsmith = None
    LANGSMITH_AVAILABLE = False

from langgraph.graph import StateGraph, END
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI

import re
from langchain_groq import ChatGroq  # Groq client (pip install langchain-groq)
import json  # needed for parsing Groq LLM JSON output


def clean_response(text: str) -> str:
    # Remove ```sql ... ``` or ``` blocks
    text = re.sub(r"```sql|```", "", text, flags=re.IGNORECASE).strip()
    return text

def extract_sql(raw_text: str) -> str | None:
    if not raw_text:
        return None

    # Remove the leading/trailing ```sql ... ``` wrappers
    cleaned = re.sub(r"^```sql\s*|```$", "", raw_text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()

    # Now find the SELECT (or other query) explicitly
    match = re.search(r"(SELECT.*;)", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # If regex fails, just return the cleaned string (better than "SELECT query.")
    return cleaned

def normalize_value(val):
    if isinstance(val, str):
        # Keep digits and 'x', strip everything else
        return re.sub(r"[^0-9x]", "", val)
    return val

def normalize_result(result):
    return [tuple(normalize_value(v) for v in row) for row in result]

# -----------------------------
# 0) FLASK + ENV + LOGGING
# -----------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
# Limit request body to ~512KB (adjust as you need)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024


load_dotenv()

# Set up LangSmith environment variables
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT") or "default"
LANGSMITH_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT") or os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com"

print(f"LANGSMITH_API_KEY: {'*' * 8 if LANGSMITH_API_KEY else 'None'}")
print(f"LANGSMITH_PROJECT: {LANGSMITH_PROJECT}")
print(f"LANGSMITH_ENDPOINT: {LANGSMITH_ENDPOINT}")
print(f"langsmith available: {LANGSMITH_AVAILABLE}")

# Configure LangSmith
if LANGSMITH_AVAILABLE and LANGSMITH_API_KEY:
    # Set environment variables for LangChain tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
    
    # Initialize LangSmith client
    try:
        langsmith_client = langsmith.Client(
            api_key=LANGSMITH_API_KEY,
            api_url=LANGSMITH_ENDPOINT
        )
        print("LangSmith client initialized successfully.")
        print("LangSmith tracing enabled.")
    except Exception as e:
        print(f"Failed to initialize LangSmith client: {e}")
        langsmith_client = None
else:
    langsmith_client = None
    print("LangSmith tracing not enabled. (Missing package or API key)")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("app")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

# Optional API key to protect /query endpoint (recommended)
API_KEY = os.getenv("API_KEY")  # If set, clients must send X-API-Key header

# DB URL can be provided either as full URL or components
DATABASE_URL = os.getenv("DATABASE_URL")

def _build_url_from_parts() -> Optional[str]:
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT") or "3306"
    name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    pwd = os.getenv("DB_PASSWORD")
    if all([host, name, user, pwd]):
        return f"mysql+mysqlconnector://{user}:{pwd}@{host}:{port}/{name}"
    return None

if not DATABASE_URL:
    DATABASE_URL = _build_url_from_parts()

if not DATABASE_URL:
    raise ValueError(
        "No database connection info. Provide DATABASE_URL or DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD in .env"
    )

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.0, max_output_tokens=1024)
# Init Groq evaluator LLM


groq_llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",   # adjust to Llama-4 once available
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0,
)

def auto_discover_schema_names() -> list[str]:
    """
    Fetch all user-defined schema (database) names from information_schema.
    Excludes system schemas like 'mysql', 'information_schema', 'performance_schema', 'sys'.
    """
    # Connect to the base DB (no specific schema)
    parts = DATABASE_URL.rsplit("/", 1)
    base_url = parts[0] if len(parts) == 2 else DATABASE_URL
    base_db = SQLDatabase.from_uri(f"{base_url}/information_schema")

    # Query for non-system schemas
    result = base_db.run("""
        SELECT SCHEMA_NAME
        FROM information_schema.SCHEMATA
        WHERE SCHEMA_NAME NOT IN (
            'mysql', 'information_schema', 'performance_schema', 'sys'
        )
        AND SCHEMA_NAME NOT LIKE 'test%'
        AND SCHEMA_NAME NOT LIKE 'tmp%'
        ORDER BY SCHEMA_NAME
    """)
    
    # SQLDatabase.run() returns a string like "[('crm',), ('onboarding',), ...]"
    # We need to parse it properly
    if isinstance(result, str):
        # Extract schema names using regex
        import ast
        try:
            # Try to safely evaluate the string as a Python literal
            rows = ast.literal_eval(result)
            return [row[0] for row in rows if row and row[0]]
        except:
            # Fallback: use regex to extract quoted strings
            import re
            matches = re.findall(r"'([^']+)'", result)
            return [m for m in matches if m not in ('mysql', 'information_schema', 'performance_schema', 'sys')]
    elif isinstance(result, list):
        return [row[0] for row in result if row and row[0]]
    else:
        log.error(f"Unexpected result type from schema query: {type(result)}")
        return []
# -----------------------------
# 1) GLOBAL DB + SCHEMA CACHE
# -----------------------------

_schema_cache: Dict[str, Tuple[float, str]] = {}  # { "default": (ts, schema_str) }

SCHEMA_CACHE_TTL_SECONDS = int(os.getenv("SCHEMA_CACHE_TTL_SECONDS", "600"))  # 10 minutes
DEFAULT_MAX_ROWS = int(os.getenv("MAX_ROWS", "10000"))  # auto-LIMIT safeguard
MAX_EXEC_MS = int(os.getenv("MAX_EXEC_MS", "8000"))   # MySQL max_execution_time in ms

SCHEMAS = auto_discover_schema_names()
log.info(f"Auto-discovered schemas: {SCHEMAS}")

# replace the single _db_singleton with a dictionary
_db_singletons: Dict[str, SQLDatabase] = {}  # key = schema name
_cross_schema_db: Optional[SQLDatabase] = None  # for cross-schema queries

# Safe mode quickly restores pre-change behavior when set to "1"
SAFE_MODE = os.getenv("SAFE_MODE", "0") == "1"
def get_db(schema: str) -> SQLDatabase:
    """
    Returns a SQLDatabase instance for the requested schema.
    Initializes it if not already done.
    """
    global _db_singletons
    if schema not in _db_singletons:
        # Build a URI pointing to the specific schema
        # If DATABASE_URL already has a database, replace it
        parts = DATABASE_URL.rsplit("/", 1)
        if len(parts) == 2:
            uri_with_schema = f"{parts[0]}/{schema}"
        else:
            uri_with_schema = f"{DATABASE_URL}/{schema}"

        _db_singletons[schema] = SQLDatabase.from_uri(
            uri_with_schema,
            sample_rows_in_table_info=0
        )
        log.info(f"Initialized SQLDatabase for schema {schema}")
    return _db_singletons[schema]

DEFAULT_SCHEMA = SCHEMAS[0] if SCHEMAS else "public"  # default to the first schema in the list
def auto_discover_schemas() -> Dict[str, list]:
    """Dynamically discover all tables per schema using information_schema."""
    schema_map = {}
    for schema in SCHEMAS:
        # 🔒 CRITICAL: Validate schema name is safe before interpolating into SQL
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", schema):
            raise ValueError(f"Invalid schema name: {schema!r}. Must match ^[a-zA-Z_][a-zA-Z0-9_]*$")
        
        db = get_db(schema)
        # Now it's safe to interpolate
        rows = db.run(f"""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = '{schema}' AND table_type = 'BASE TABLE'
        """)
        schema_map[schema] = [row[0] for row in rows]
    return schema_map

# Replace hardcoded SCHEMA_TABLES
SCHEMA_TABLES = auto_discover_schemas()

def extract_schemas_from_sql(sql: str) -> list[str]:
    print("🔍 Using sqlglot-based schema extractor")
    try:
        parsed = sqlglot.parse_one(sql, dialect="mysql")
        schemas = set()
        for table in parsed.find_all(exp.Table):
            if table.db:  # schema.table
                schema = table.db
                if schema in SCHEMAS:
                    schemas.add(schema)
            else:  # just table
                table_name = table.name
                for sch, tables in SCHEMA_TABLES.items():
                    if table_name in tables:
                        schemas.add(sch)
                        break
        result = list(schemas) or [DEFAULT_SCHEMA]
        print(f"✅ Detected schemas: {result}")
        return result
    except Exception as e:
        log.warning(f"sqlglot parsing failed: {e}")
        return [DEFAULT_SCHEMA]
    
def extract_schema_from_sql(sql: str) -> str:
    schemas = extract_schemas_from_sql(sql)
    return schemas[0] if schemas else DEFAULT_SCHEMA


def get_cross_schema_db() -> SQLDatabase:
    """
    Returns a SQLDatabase instance that can access multiple schemas.
    This is used for cross-schema queries.
    """
    global _cross_schema_db
    if _cross_schema_db is None:
        # For cross-schema queries, we need to connect to a database that can access multiple schemas
        # We'll use the first schema as the default connection point, but the SQL queries will use
        # fully qualified table names (schema.table) to access other schemas
        schema = DEFAULT_SCHEMA
        db = get_db(schema)
        _cross_schema_db = db
        log.info(f"Initialized cross-schema SQLDatabase using {schema} schema")
    return _cross_schema_db

def get_schema_cached() -> str:
    now = time.time()
    entry = _schema_cache.get("default")
    if entry and (now - entry[0] < SCHEMA_CACHE_TTL_SECONDS):
        return entry[1]

    combined_schema = []
    for schema_name in SCHEMAS:
        db = get_db(schema_name)  # database for this schema
        info = db.get_table_info()  # ❌ do NOT pass schema=
        combined_schema.append(f"-- Schema: {schema_name}\n{info}")

    full_schema = "\n\n".join(combined_schema)
    _schema_cache["default"] = (now, full_schema)
    return full_schema

def get_cross_schema_info() -> str:
    """
    Returns schema information for cross-schema queries.
    This includes all tables from all schemas with their schema prefixes.
    """
    now = time.time()
    entry = _schema_cache.get("cross_schema")
    if entry and (now - entry[0] < SCHEMA_CACHE_TTL_SECONDS):
        return entry[1]

    combined_schema = []
    for schema_name in SCHEMAS:
        db = get_db(schema_name)
        info = db.get_table_info()
        # Add schema prefix to table names in the schema info
        # This helps the LLM understand which schema each table belongs to
        schema_prefixed_info = re.sub(
            r'CREATE TABLE `?([a-zA-Z_][a-zA-Z0-9_]*)`?',
            f'CREATE TABLE `{schema_name}`.`\\1`',
            info
        )
        combined_schema.append(f"-- Schema: {schema_name}\n{schema_prefixed_info}")

    full_schema = "\n\n".join(combined_schema)
    _schema_cache["cross_schema"] = (now, full_schema)
    return full_schema

# -----------------------------
# 2) PII REDACTION + HELPERS
# -----------------------------
sensitive_patterns = {
    "phone": ["phone", "mobile", "telephone", "cell", "contact_number", "phone_number", "tel"],
    "dob": ["dob", "date_of_birth", "birth_date", "birth", "birthday", "birthdate"],
    "address": ["address", "street", "city", "state", "province", "zip", "zipcode", "postal_code", "country", "location", "addr"],
    "ssn": ["ssn", "social_security", "social_security_number", "socialsecuritynumber"],
    "pan": ["pan", "pan_number", "pancard", "permanent_account_number"],
    "aadhaar": ["aadhaar", "aadhar", "uidai", "aadhaar_number"],
    "passport": ["passport", "passport_number", "passport_no"],
    "credit_card": ["credit_card", "card_number", "cc_number", "creditcard", "card_num", "cvv", "cvc", "expiry", "exp_date"],
    "bank": ["bank", "account_number", "iban", "swift", "routing", "bank_name", "ifsc"],
    "gender": ["gender", "sex", "pronoun"],
    "tax_id": ["tax_id", "tin", "tax_identification", "gst", "vat_number", "ein"],
    "username": ["username", "user_id", "login", "handle", "screen_name"],
    "password": ["password", "pass", "pwd", "secret", "credential", "hash"],
    "ip_address": ["ip_address", "ip_addr", "ipv4", "ipv6"],
    "device_id": ["device_id", "imei", "mac_address", "udid", "serial_number"],
    "national_id": ["national_id", "id_number", "govt_id", "citizen_id"],
}
def get_sensitivity_type(col_name: str) -> Optional[str]:
    """
    Returns the PII type (e.g., 'phone', 'dob') if column is sensitive, else None.
    """
    col = col_name.lower()
    for key, patterns in sensitive_patterns.items():
        if any(p in col for p in patterns):
            return key
    return None
# Match a likely column definition row inside CREATE TABLE (...) block.
# Skips lines that start with constraints/keys.
DDL_COL_LINE = re.compile(
    r"^\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s+[A-Za-z]+(?:\s*\([^\)]*\))?(?:\s+(?:UNSIGNED|NOT NULL|NULL|DEFAULT|AUTO_INCREMENT|COMMENT|CHARSET|COLLATE|PRIMARY|UNIQUE|KEY|CHECK).*)?,?\s*$",
    re.IGNORECASE
)

DDL_SKIP_PREFIXES = (
    "primary key", "unique key", "key ", "constraint", "foreign key", ") ENGINE", ")", "/*", "--"
)

def anonymize_schema(schema: str) -> Tuple[str, Dict[str, str]]:
    """
    Redacts column names with DESCRIPTIVE placeholders to guide the LLM.
    mapping: {placeholder -> real_column}
    """
    sanitized_lines = []
    pii_mapping: Dict[str, str] = {}
    # Keep track of counts for each keyword (e.g., phone_1, phone_2)
    redaction_counts: Dict[str, int] = {} 

    for raw_line in schema.splitlines():
        line = raw_line.rstrip("\n")
        lstrip = line.lstrip().lower()
        if any(lstrip.startswith(prefix) for prefix in DDL_SKIP_PREFIXES):
            sanitized_lines.append(line)
            continue

        m = DDL_COL_LINE.match(line)
        if m:
            colname = m.group(1)
            sensitivity_type = get_sensitivity_type(colname)
            if sensitivity_type:
                count = redaction_counts.get(sensitivity_type, 0) + 1
                redaction_counts[sensitivity_type] = count
                placeholder = f"redacted_{sensitivity_type}_{count}"
                replaced = re.sub(
                    rf"(^\s*)`?{re.escape(colname)}`?",
                    rf"\1`{placeholder}`",
                    line,
                    count=1,
                    flags=re.IGNORECASE
                )
                pii_mapping[placeholder] = colname
                sanitized_lines.append(replaced)
                continue
        sanitized_lines.append(line)
    return "\n".join(sanitized_lines), pii_mapping

def deanonymize_query(sql: str, mapping: Dict[str, str]) -> str:
    """
    Replace placeholders with real column names. Works with or without backticks.
    """
    for placeholder, real in mapping.items():
        # Replace occurrences as identifiers, optionally wrapped in backticks.
        sql = re.sub(
            rf"(?<![A-Za-z0-9_`])`?{re.escape(placeholder)}`?(?![A-Za-z0-9_`])",
            f"`{real}`",
            sql
        )
    return sql

# -----------------------------
# 3) SQL SAFETY / VALIDATION
# -----------------------------
DISALLOWED_TOKENS = [
    # Block only multi-statement separators and dangerous keywords
    "insert", "update", "delete", "drop", "alter", "truncate", "create", "grant",
    "revoke", "rename", "replace", "call", "use", "show", "set", "describe"
]

DISALLOWED_SEMICOLONS = 2  # allow one semicolon at the end

def contains_disallowed(query: str) -> bool:
    """
    Returns True if any dangerous keywords appear as whole words in the SQL.
    """
    lc = query.lower()
    for tok in DISALLOWED_TOKENS:
        if re.search(rf"\b{tok}\b", lc):
            return True
    return False

def parse_groq_json(resp_text: str) -> dict:
    # Extract first JSON-looking block
    match = re.search(r"\{.*\}", resp_text, flags=re.DOTALL)
    if not match:
        return {"error": "Failed to parse Groq eval output", "raw": resp_text}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {"error": "Failed to parse Groq eval output", "raw": resp_text}
    
def enforce_limit(query: str, max_rows: int = DEFAULT_MAX_ROWS) -> str:
    """
    If no LIMIT is present (outside subqueries heuristic), append a LIMIT.
    """
    lc = query.lower()
    # crude check—if top-level already has limit, keep it
    if re.search(r"\blimit\s+\d+", lc):
        return query
    return query.rstrip().rstrip(";") + f" LIMIT {max_rows}"

def validate_sql_or_error(db: SQLDatabase, sql: str) -> Optional[str]:
    """
    Returns None if OK, otherwise an error string safe to show to user.
    Uses EXPLAIN to validate syntax/plan (fast).
    """

    lowered = sql.strip().lower()

    # 1. Block multiple statements (split by ;)
    # 1. Block multiple top-level statements
    if sql.strip().count(";") > 1:
        return "Only a single SQL statement is allowed."

    if not (lowered.startswith("select") or lowered.startswith("with")):
        return "Only read-only SELECT queries are allowed."

    # 3. Block disallowed keywords (word-boundaries to avoid false positives like created_at)
    disallowed = [
        "insert", "update", "delete", "drop", "alter", "truncate",
        "create", "grant", "revoke", "outfile", "infile"
    ]
    for tok in disallowed:
        if re.search(rf"\b{tok}\b", lowered):
            return "Disallowed keyword detected in query."

    # 4. Try EXPLAIN to validate correctness
    try:
        db.run(f"EXPLAIN {sql}")
        return None
    except Exception:
        return "Your query appears invalid or references non-existent tables/columns."


def sql_references_table(sql: str) -> bool:
    """
    Heuristic check whether the SQL references at least one table from the known schemas.
    Returns True if it finds a FROM/JOIN, a fully-qualified schema.table reference, or
    any table name present in SCHEMA_TABLES. This helps catch cases where the model
    returns a string literal (e.g. SELECT '...') instead of a real query.
    """
    if not sql:
        return False

    # Quick checks for FROM / JOIN clauses
    if re.search(r"\bfrom\b|\bjoin\b", sql, flags=re.IGNORECASE):
        return True

    # Fully-qualified schema.table references
    if re.search(r"\b[a-zA-Z_][\w]*\.[a-zA-Z_][\w]*\b", sql):
        return True

    # Check known table names across schemas
    for tables in SCHEMA_TABLES.values():
        for t in tables:
            if re.search(rf"\b{re.escape(t)}\b", sql, flags=re.IGNORECASE):
                return True

    return False

# -----------------------------
# 4) DOMAIN HINTS & BUSINESS CONTEXT
# -----------------------------

# Comprehensive domain hints for ZeeProc CSM platform
DOMAIN_HINTS_TEXT = """
## Schema-to-Domain Mapping
- **zeeproc**: Core platform data for customer success management, including master customer data (zeeproc.mastercustomertable), agent run metadata (zeeproc.agent_run_metadata), communications (zeeproc.communication_log), and client goals (zeeproc.client_goals)
- **crm**: Customer Relationship Management (external CRM integration, customer data) such as crm.crm_account, crm.crm_contact, crm.crm_deal
- **sm**: Service Management (communication logs, email management, support tickets) such as sm.sm_account, sm.sm_contact, sm.sm_case
- **onboarding**: Customer onboarding workflows, milestones, tasks, and outcomes like onboarding.onboarding_plan, onboarding.plan_milestone, onboarding.plan_task
- **pat**: Platform Analytics & Tracking (agent performance, metrics, automation logs) including pat.pat_user, pat.pat_daily_metrics, pat.pat_engagement_metrics

## Business Term Mappings
- "customer name" → zeeproc.mastercustomertable.customer_company_name
- "customer id" → zeeproc.mastercustomertable.customer_id
- "account name" → crm.crm_account.account_name
- "deal amount" → crm.crm_deal.amount
- "case status" → sm.sm_case.status
- "plan status" → onboarding.onboarding_plan.status
- "task name" → onboarding.plan_task.name
- "agent name" → zeeproc.agent_run_metadata.agent_name
- "metric value" → pat.pat_engagement_metrics.events_count

### Customer Health & Status
- "customer health" → Look in crm.crm_account, sm.sm_case_metrics, onboarding.onboarding_plan, zeeproc.mastercustomertable
- "customer status" → Check crm.crm_account, onboarding.onboarding_plan.status, zeeproc.client_goals.task_status
- "overdue" → Look for dates < current_date in onboarding.plan_milestone, onboarding.plan_task
- "at-risk customers" → Check pat.pat_engagement_metrics, sm.sm_case_metrics, onboarding.onboarding_plan delays, zeeproc.agent_execution_logs failures
- "renewal dates" → Look in crm.crm_deal.close_date, crm.crm_financial_metrics
- "satisfaction scores" → Check zeeproc.client_goals, sm.sm_case_metrics

### Revenue & Financial Analytics
- "ARR" (Annual Recurring Revenue) → crm.crm_financial_metrics.total_mrr
- "revenue" → crm.crm_deal.amount, crm.crm_financial_metrics.total_revenue_current_value
- "upsell opportunities" → Check crm.crm_product, crm.crm_deal.deal_stage, pat.pat_daily_metrics usage patterns
- "churn risk" → Analyze low values in pat.pat_engagement_metrics.events_count or session_duration over time, sm.sm_case frequency, onboarding.onboarding_plan delays
- "financial metrics" → crm.crm_financial_metrics, crm.crm_order.total_amount

### Agent Performance & Automation
- "agent performance" → zeeproc.agent_execution_logs, zeeproc.agent_run_metadata, zeeproc.agent_execution_functional_metrics
- "automated tasks" → zeeproc.agent_run_metadata, zeeproc.agent_execution_details
- "AI agents" → zeeproc.agent_execution_logs, zeeproc.agent_prompts, zeeproc.agent_run_metadata
- "agent success rates" → zeeproc.agent_run_metadata.is_completed, zeeproc.agent_execution_logs success rates
- "agent failures" → zeeproc.agent_execution_logs with error conditions, zeeproc.agent_notification

### Communication & Engagement
- "emails" → sm.sm_case, zeeproc.communication_log, zeeproc.communication_detail
- "unread emails" → zeeproc.communication_log.state = 'UNREAD' 'UNREAD' word should be in caps lock.
- "communication logs" → sm.sm_case, zeeproc.communication_log, zeeproc.communication_detail
- "response time" → Calculate from zeeproc.communication_log.created_at timestamps
- "sentiment" → Check zeeproc.communication_detail.content analysis
- "support tickets" → sm.sm_case, sm.sm_case_metrics

### Onboarding & Project Management
- "onboarding" → onboarding.onboarding_plan, onboarding.plan_milestone, onboarding.plan_task
- "milestones" → onboarding.plan_milestone, onboarding.milestone_goal
- "onboarding progress" → Check onboarding.plan_milestone.status, onboarding.plan_task.is_completed
- "behind schedule" → Compare onboarding.plan_milestone.actual_end_date vs end_date, onboarding.plan_task.actual_end_date vs end_date
- "completion rates" → Calculate from onboarding.plan_milestone.status, onboarding.plan_task.is_completed
- "time-to-value" → Measure from onboarding.onboarding_plan.created_at to first completed milestone

### Team & Workload Management
- "CSM workload" → Count active customers per CSM in zeeproc.mastercustomertable, onboarding.plan_task assignments
- "workload distribution" → Analyze onboarding.plan_task.owner_name, onboarding.plan_stakeholder assignments
- "overdue tasks" → Check onboarding.plan_task where end_date < current_date and is_completed = false
- "team performance" → Compare completion rates across onboarding.plan_task.owner_name, onboarding.plan_stakeholder.name

## Key Entity Relationships
- **Customer Data Flow**: zeeproc.mastercustometable → (crm.crm_account, onboarding.onboarding_plan, sm.sm_account)
- **Communication Flow**: zeeproc.communication_log → zeeproc.communication_detail → sm.sm_case
- **Agent Execution**: zeeproc.agent_run_metadata → zeeproc.agent_run_details → zeeproc.agent_execution_logs
- **Onboarding Flow**: onboarding.onboarding_plan → onboarding.plan_milestone → onboarding.plan_task
- **CRM Integration**: crm.crm_account → crm.crm_contact → crm.crm_deal → crm.crm_activity

## Common Query Patterns
- Cross-schema queries often join: zeeproc.mastercustomertable + crm.crm_account + onboarding.onboarding_plan + sm.sm_account
- Agent performance queries typically involve: zeeproc.agent_run_metadata + zeeproc.agent_execution_logs + zeeproc.agent_execution_functional_metrics
- Customer health queries combine: crm.crm_account + zeeproc.communication_log + onboarding.onboarding_plan + pat.pat_engagement_metrics
- Revenue queries focus on: crm.crm_deal + crm.crm_financial_metrics + crm.crm_order

## Important Notes
- Always use fully qualified table names (schema.table) for cross-schema queries
- Customer data spans multiple schemas - use customer_id for joins across zeeproc.mastercustomertable, crm.crm_account, onboarding.onboarding_plan, sm.sm_account, pat.pat_user
- Agent execution data is primarily in zeeproc schema (agent_run_metadata, agent_execution_logs, agent_prompts)
- Communication data is in zeeproc schema (communication_log, communication_detail)
- Service management data is in sm schema (sm_account, sm_contact, sm_case)
- Onboarding workflows are in onboarding schema
- CRM data is in crm schema
- Analytics data is in pat schema

## Common Table Patterns by Domain

### Customer Management
- zeeproc.mastercustomertable, crm.crm_account, crm.crm_contact, crm.crm_deal

### Communication & Support
- zeeproc.communication_log, zeeproc.communication_detail, sm.sm_accounts, sm.sm_contacts, sm.sm_case, sm.sm_case_metrics

### Onboarding & Project Management  
- onboarding.plan_template, onboarding.onboarding_plan, onboarding.plan_milestone, onboarding.plan_task
- onboarding.plan_outcome, onboarding.milestone_goal, onboarding.plan_stakeholder, onboarding.plan_feedback_contact

### Agent Performance & Analytics
- zeeproc.agent_run_metadata, zeeproc.agent_run_details, zeeproc.agent_prompts, zeeproc.agent_execution_logs
- zeeproc.agent_execution_details, zeeproc.agent_notification, zeeproc.agent_execution_functional_metrics

### Platform Analytics & Tracking
- pat.pat_user, pat.pat_daily_metrics, pat.pat_summary_metrics, pat.pat_engagement_metrics

### Financial & Revenue
- crm.crm_financial_metrics, crm.crm_deal, crm.crm_order, crm.crm_product
- Look for revenue-related fields: amount, value, total_amount, total_mrr, total_revenue_current_value

### CRM Integration & OAuth
- crm.crm_oauth_token, crm.crm_oauth_session, crm.sync_log


## Table Descriptions

### Zeeproc Schema
- MasterCustomerTable: Core customer info with company details and linked account IDs.
- Agent_Run_Metadata: Metadata on agent runs tracking customer and client info.
- Agent_Run_Details: Detailed input/output data for each agent run.
- Agent_Prompts: Defines agent prompts including text and versioning.
- Client_Goals: Business goals set for clients with status tracking.
- Communication_Log: Logs customer interactions with channels and message states.
- Communication_Detail: Stores communication event content linked to logs.
- Agent_Execution_Logs: Execution logs of software agents with timing and status.
- Agent_Execution_Details: Input/output JSON linked to agent execution logs.
- Agent_Notification: Notifications from agents targeted at users.
- Agent_Execution_Functional_Metrics: Performance metrics collected during executions.

### SM Schema
- sm_accounts: Social media accounts linked to customers.
- sm_contacts: Contacts related to social media accounts.
- sm_case: Support or issue cases linked to social media accounts and contacts.
- sm_case_metrics: Aggregated metrics on social media case performance.

### PAT Schema
- pat_users: Users with email, domain, and segment data.
- pat_daily_metrics: Daily behavior and usage metrics.
- pat_summary_metrics: Periodic aggregate metrics for usage trends.
- pat_engagement_metrics: Detailed engagement event counts and durations.

### Onboarding Schema
- plan_template: Onboarding plan templates.
- onboarding_plan: Customer onboarding plans tracking status and stages.
- plan_project_detail: Financial and scheduling details of onboarding.
- plan_stakeholder: Contacts and roles involved in onboarding.
- plan_feedback_contact: Specific feedback contacts in onboarding.
- plan_milestone: Key onboarding milestones with dates and status.
- milestone_goal: Goals defined per milestone.
- milestone_feedback_contact: Contacts for milestone feedback.
- milestone_comment: Notes and comments on milestones.
- plan_task: Tasks assigned to milestones with owners and dates.
- plan_outcome: Expected results from onboarding plans.

### CRM Schema
- crm_oauth_token: OAuth tokens for CRM platform integration.
- crm_oauth_session: OAuth authorization session data.
- crm_account: CRM account info including financials and industry.
- crm_contact: Contacts within CRM accounts.
- crm_deal: Sales deals with financial and status info.
- crm_activity: Activities related to CRM contacts and accounts.
- crm_product: Products linked to sales deals.
- crm_order: Orders placed within the CRM system.
- sync_log: Logs of integration sync operations.
- crm_financial_metrics: Financial KPIs and churn analytics.

## Critical Column Name Mappings
- “customer name” → zeeproc.master_customer_table.customer_company_name  
- “customer id” → zeeproc.master_customer_table.customer_id  
- “account name” → crm.crm_account.account_name  
- “deal amount” → crm.crm_deal.amount  
- “case status” → sm.sm_case.status  
- “plan status” → onboarding.onboarding_plan.status  
- “task name” → onboarding.plan_task.name  
- “agent name” → zeeproc.agent_run_metadata.agent_name  
- “metric value” → pat.pat_engagement_metrics.events_count  

## Key Schema Relationships
- Customer → SM: mastercustomertable.sm_account_id = sm_account.sm_account_id = sm_case.sm_account_id
- Customer → CRM: mastercustomertable.crm_account_id = crm_account.crm_account_id
- Customer → Onboarding: mastercustomertable.customer_id = onboarding_plan.customer_id
- Plans → Milestones → Tasks: onboarding_plan.plan_id = plan_milestone.plan_id = plan_task.milestone_id
- CSM assignments: agent_run_metadata.csm_name
- Overdue = end_date < CURDATE() AND status <> 'completed'
- Use FALSE/TRUE for booleans, not 0/1


### Join Validation Rules
- Never join tables directly on customer_id unless explicitly shown in relationships above
- Always use intermediate tables (e.g., sm_account) when connecting schemas
- Verify all column names exist in the provided schema before using them

"""

def retrieve_context_heuristic(question: str) -> Tuple[str, str]:
    """
    Fast, regex/keyword-based retrieval of relevant schema + domain hints.
    Returns (minimal_schema_ddl, filtered_domain_hints)
    """
    q_lower = question.lower()
    
    # --- Schema keyword mapping ---
    SCHEMA_KEYWORDS = {
        "crm": ["customer", "account", "deal", "revenue", "financial", "crm", "contact", "industry", "arr", "mrr", "churn", "health"],
        "onboarding": ["onboarding", "milestone", "plan", "task", "behind", "overdue", "schedule", "completion", "progress", "kickoff", "timeline", "delay"],
        "zeeproc": ["agent", "automation", "execution", "prompt", "communication", "email", "notification", "csm", "workload", "goal", "client", "master"],
        "sm": ["case", "support", "ticket", "email", "communication", "response", "sentiment", "sm", "social"],
        "pat": ["metric", "engagement", "usage", "analytics", "performance", "activity", "daily", "summary", "pat", "decline", "trend", "event", "health"]
    }
    BUSINESS_TERM_MAPPINGS = {
    "customer health": ["crm", "sm", "onboarding", "zeeproc"],
    "at-risk customers": ["pat", "sm", "onboarding", "zeeproc"],
    "CSM workload": ["zeeproc", "onboarding"],
    "unread emails": ["zeeproc"],
    "response time": ["zeeproc"],
    "churn risk": ["pat", "sm", "onboarding"],
    }
   
    relevant_schemas = set()

    # 1. Exact schema name match
    for schema in SCHEMAS:
        if schema in q_lower:
            relevant_schemas.add(schema)

    # 2. Business-term mappings (high-value signals)
    for term, schemas in BUSINESS_TERM_MAPPINGS.items():
        if term in q_lower:
            relevant_schemas.update(schemas)

    # 3. Generic keyword fallback (always run)
    for schema, keywords in SCHEMA_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            relevant_schemas.add(schema)

    # 4. Multi-schema → include golden key
    if len(relevant_schemas) >= 2:
        relevant_schemas.add("zeeproc")

    # 5. Ultimate fallback
    if not relevant_schemas:
        log.warning(f"No schema detected for question: {question}")
        # If no relevant schemas detected, return empty schema to signal upstream
        # that this question is unrelated to the DB. Caller should handle this case
        # and avoid running SQL generation/execution.
        return "", "No relevant schema detected."

    # 2. Build minimal schema DDL
    schema_lines = []
    for schema in relevant_schemas:
        db = get_db(schema)
        info = db.get_table_info()
        prefixed = re.sub(
            r'CREATE TABLE `?([a-zA-Z_][a-zA-Z0-9_]*)`?',
            f'CREATE TABLE `{schema}`.`\\1`',
            info
        )
        schema_lines.append(f"-- Schema: {schema}\n{prefixed}")
    minimal_schema = "\n".join(schema_lines)

    # 3. Filter DOMAIN_HINTS_TEXT by section (not line) to avoid noise
    all_keywords = set()
    for s in relevant_schemas:
        all_keywords.update(SCHEMA_KEYWORDS.get(s, []))
    all_keywords.update(relevant_schemas)  # e.g., "crm", "pat"

    # Split by section headers (lines starting with "## ")
    sections = re.split(r"(?=\n## )", DOMAIN_HINTS_TEXT.strip())
    relevant_sections = []

    for section in sections:
        if any(kw in section.lower() for kw in all_keywords):
            relevant_sections.append(section)

    filtered_domain_hints = "".join(relevant_sections).strip() or "No specific domain hints."
    return minimal_schema, filtered_domain_hints
# -----------------------------
# 5) LLM PROMPTS
# -----------------------------
SQL_GENERATION_PROMPT = """### Task
You are a world-class SQL developer specializing in Customer Success Management (CSM) platforms. Generate a single, syntactically correct MySQL SELECT query to answer the user's question, based only on the provided database schema.

### Context
You are working with ZeeProc, an Agentic AI Customer Success Management platform that automates the entire customer lifecycle from onboarding to renewal. The platform manages:
- Customer relationships and CRM data
- Communication logs and email management  
- Onboarding workflows and milestones
- AI agent performance and automation
- Team workload and productivity metrics

### Rules
- Output ONLY the SQL query (no backticks, no commentary).
- Use ONLY tables/columns from the schema.
- Do NOT include comments.
- You must generate exactly one complete SQL statement. 
  - Subqueries (nested SELECTs) are allowed.
  - UNION, INTERSECT, and EXCEPT are allowed.
  - Window functions (e.g., ROW_NUMBER, RANK, DENSE_RANK) are allowed.
- Queries must be read-only (no INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, etc.).
- Do NOT use multiple separate statements separated by semicolons.
- Aggregates, filters, sorting (ORDER BY), and limiting (LIMIT) are allowed.
- Use explicit JOINs when combining tables.
- For cross-schema queries, use fully qualified table names (schema.table) when referencing tables from different schemas.
- Pay special attention to customer_id for joining data across schemas.
- Consider business context: customer health, revenue metrics, agent performance, communication patterns, onboarding progress.

### Literal Value Rules
- Use lowercase for all string literals in comparisons (e.g., 'active', 'completed', 'in_progress', 'unread').
- Do NOT capitalize status or category values.
- Example:
  “WHERE pm.status <> 'completed'”  ✅
  “WHERE pm.status <> 'Completed'”  ❌

### Domain Hints & Business Context
{domain_hints}

### Database Schema
{schema}

### User Question
{question}

### SQL Query
"""

RESULT_SYNTHESIS_PROMPT = """### Task
You are a helpful assistant. Provide a concise, natural language answer to the user's question based on the SQL result rows provided.

### Rules
- If the result is empty, state that no matching data was found.
- Summarize clearly; do not reveal SQL.

### Original Question
{question}

### Database Query Result (rows)
{result}

### Your Answer
"""

# -----------------------------
# 6) LANGGRAPH STATE & NODES
# -----------------------------
class GraphState(TypedDict):
    question: str
    db_schema: str
    pii_mapping: Dict[str, str]
    filtered_domain_hints: str  # <-- ADD THIS
    sanitized_schema: str
    anonymized_sql_query: str
    executable_sql_query: str
    error_message: str
    result: Any
    final_answer: str


def anonymize_schema_node(state: GraphState) -> GraphState:
    sanitized, mapping = anonymize_schema(state["db_schema"])
    return {"sanitized_schema": sanitized, "pii_mapping": mapping, "error_message": ""}

def generate_sql_node(state: GraphState) -> GraphState:
    domain_hints = state["filtered_domain_hints"]  
    prompt = SQL_GENERATION_PROMPT.format(
        schema=state["sanitized_schema"],
        question=state["question"],
        domain_hints=domain_hints,
    )
    response = llm.invoke(prompt)
    print("RAW LLM RESPONSE:", response)  # <-- Debug line
    sql = response.content.strip()
    cleaned = clean_response(response.content)
    sql = extract_sql(cleaned) or cleaned
    print("CLEANED SQL:", sql)  # <-- Debug line
    # Basic checks now; full validation later
    if not sql or not sql.lower().startswith("select"):
        return {"error_message": "The model did not generate a valid SELECT query."}

    return {"anonymized_sql_query": sql, "error_message": ""}

def deanonymize_sql_node(state: GraphState) -> GraphState:
    query = deanonymize_query(state["anonymized_sql_query"], state["pii_mapping"])
    # Enforce LIMIT as a safety net
    query = enforce_limit(query, DEFAULT_MAX_ROWS)
    return {"executable_sql_query": query, "error_message": ""}

def sql_validator_node(state: GraphState) -> GraphState:
    user_sql = state["executable_sql_query"]

    # Check if this is a cross-schema query
    schemas = extract_schemas_from_sql(user_sql)
    is_cross_schema = len(schemas) > 1
    
    if is_cross_schema:
        # Use cross-schema database for validation
        db = get_cross_schema_db()
    else:
        # Use single schema database
        schema = schemas[0] if schemas else DEFAULT_SCHEMA
        db = get_db(schema)

    print("FINAL QUERY SENT TO VALIDATOR:", repr(user_sql))
    print(f"Using {'cross-schema' if is_cross_schema else 'single-schema'} database for validation")# 0) Ensure the SQL actually references schema tables (not a plain string literal)
    if not sql_references_table(user_sql):
        return {"error_message": "Generated SQL does not reference any table from the provided schema. The model appears to have returned a literal or non-database response. Please generate a query that selects from the schema tables."}

    # 1) Validate syntax/plan via EXPLAIN
    err = validate_sql_or_error(db, user_sql)
    if err:
        return {"error_message": f"SQL Validation Error: {err}"}

    return {}

def execute_sql_node(state: GraphState) -> GraphState:
    sql = state["executable_sql_query"]
    
    # Check if this is a cross-schema query
    schemas = extract_schemas_from_sql(sql)
    is_cross_schema = len(schemas) > 1
    
    if is_cross_schema:
        # Use cross-schema database for execution
        db = get_cross_schema_db()
    else:
        # Use single schema database
        schema = schemas[0] if schemas else DEFAULT_SCHEMA
        db = get_db(schema)

    try:
        # Harden the session a bit (read-only + timeout). Some MySQL variants may not support both; ignore failures.
        try:
            db.run("SET SESSION TRANSACTION READ ONLY")
        except Exception:
            pass
        try:
            db.run(f"SET SESSION max_execution_time={MAX_EXEC_MS}")
        except Exception:
            pass

        result = db.run(sql)
        print("SQL EXECUTION RESULT:", result)  # <-- Debug line
        print(f"Executed using {'cross-schema' if is_cross_schema else 'single-schema'} database")
        return {
            "result": result,
            "executable_sql_query": sql,   # <-- forward SQL query explicitly
            "error_message": ""
        }
    except Exception as e:
        log.exception("Database Execution Error")
        return {"error_message": "Database Execution Error. Please refine your question."}

def result_synthesizer_node(state: GraphState) -> GraphState:
    # Ensure 'result' exists
    result_rows = state.get("result", [])
    
    # Format DB rows as readable text for LLM
    result_text = "\n".join([", ".join(map(str, row)) for row in result_rows])
    
    prompt = RESULT_SYNTHESIS_PROMPT.format(
        question=state["question"],
        result=result_text
    )
    
    response = llm.invoke(prompt)
    
    return {
        **state,  # keep everything else
        "final_answer": response.content.strip(),
        "error_message": ""
    }


def handle_error_node(state: GraphState) -> GraphState:
    # Do NOT leak internals; state["error_message"] is already user-safe
    error = state.get("error_message") or "An unknown error occurred."
    return {"final_answer": f"I couldn’t complete that request: {error}"}

def route_after_generation(state: GraphState) -> str:
    return "handle_error" if state.get("error_message") else "de_anonymizer"

def route_after_validation(state: GraphState) -> str:
    return "handle_error" if state.get("error_message") else "sql_executor"

def route_after_execution(state: GraphState) -> str:
    return "handle_error" if state.get("error_message") else "result_synthesizer"

def structural_eval(sql: str) -> dict:
    """
    Lightweight regex/rule-based checks for SQL safety and structure.
    Allows nested SELECTs, UNIONs, CTEs, and window functions.
    """
    errors = []
    sql_stripped = sql.strip()
    sql_lower = sql_stripped.lower()

    # Must start with SELECT or WITH (for CTEs)
    if not sql_lower.startswith(("select", "with")):
        errors.append("Query does not start with SELECT or WITH.")

    # Count top-level semicolons; allow only 1 at the end
    semicolon_count = sql_stripped.count(";")
    if semicolon_count > 1:
        errors.append("Multiple separate statements are not allowed.")
    
    # Recommend LIMIT for safety, but not mandatory if using window functions
    if not re.search(r"\blimit\s+\d+", sql_lower):
        errors.append("Query missing LIMIT clause (recommended for safety).")

    # Disallowed keywords
    if contains_disallowed(sql):
        errors.append("Query contains disallowed keywords (DML/DDL not allowed).")

    return {
        "structural_errors": errors,
        "passed": len(errors) == 0
    }

def groq_eval(question: str, schema: str, sql: str, answer: str, result: list) -> Dict[str, Any]:
    """
    Use Groq LLM to grade correctness/helpfulness/safety based on actual DB result.
    """
    # Convert raw query result to readable string
    ground_truth = "\n".join([", ".join([str(col) for col in row]) for row in result]) if result else "NO RESULTS"

    prompt = f"""
You are an expert SQL and data evaluator. Use ONLY the provided schema and the explicit database result to judge whether the generated SQL and the natural-language answer correctly and logically answer the user's question.

Context:
- User question:
{question}

- Database schema (full cross-schema context):
{schema}

- Generated SQL:
{sql}

- Database result (the actual rows returned by executing the SQL):
{ground_truth}

- Natural-language answer produced for the user:
{answer}

Instructions (strict):
1) First determine whether the SQL query is logically correct for the user's question given the schema. "Logically correct" means the query uses appropriate tables/columns/joins/filters/aggregations to answer the question. If the SQL does not reference any table from the schema (for example, it is a SELECT of a literal string), mark SQL relevance as 1 (very poor) and explain why.
2) If the SQL runs but uses wrong joins, wrong filters, wrong grouping/aggregation, or returns unrelated columns, mark SQL relevance low and explain the logical error explicitly (e.g., "uses table X when table Y should be used", "missing join on customer_id", "aggregates without GROUP BY").
3) Compare the database result to the natural-language answer. If the answer does not accurately reflect the explicit database result, mark answer_helpfulness appropriately (1-5) and explain the mismatch. Do not infer beyond the explicit rows.
4) Evaluate safety: detect PII exposure or policy violations in the SQL, result, or answer. Use 1-5 scale and call out the issue.
5) Do NOT rewrite the SQL or the answer. Use only the given schema and result. Do NOT consult outside knowledge except to identify common SQL anti-patterns.

Output: Respond in strict JSON (no surrounding text) with the following keys:
{{
    "sql_relevance": "<1-5 integer>",
    "sql_relevance_reason": "<short string explaining logical issues or why it is correct>",
    "answer_helpfulness": "<1-5 integer>",
    "answer_helpfulness_reason": "<short string>",
    "safety": "<1-5 integer>",
    "safety_reason": "<short string if any>",
    "overall_comment": "<concise summary>"
}}

Examples of reasons: "No table referenced; query returns a literal string instead of selecting from schema tables.", "Missing join on customer_id between crm and zeeproc.", "Aggregates used without GROUP BY; likely incorrect.", "Answer matches database result exactly."
"""

    resp = groq_llm.invoke(prompt)
    try:
        parsed = parse_groq_json(resp.content)
    except Exception:
        # Strip Markdown code blocks and try again
        cleaned = re.sub(r"```.*?```", "", resp.content, flags=re.DOTALL).strip()
        try:
            parsed = parse_groq_json(cleaned)
        except Exception:
            parsed = {"error": "Failed to parse Groq eval output", "raw": resp.content}
    return parsed



def run_llm_evals(question: str, schema: str, final_state: Dict[str, Any]) -> Dict[str, Any]:
    sql = final_state.get("executable_sql_query", "")
    answer = final_state.get("final_answer", "")
    result = final_state.get("result", [])  # <--- pass the actual DB query result

    # 1) Structural evals (regex-based)
    structural = structural_eval(sql)

    # 2) Groq evals (only if structural passes)
    groq_result = {}
    if structural["passed"]:
        groq_result = groq_eval(question, schema, sql, answer, result)  # <--- pass result here

    return {
        "structural": structural,
        "groq_eval": groq_result,
    }

# Assemble LangGraph
workflow = StateGraph(GraphState)
workflow.add_node("anonymizer", anonymize_schema_node)
workflow.add_node("sql_generator", generate_sql_node)
workflow.add_node("de_anonymizer", deanonymize_sql_node)
workflow.add_node("sql_validator", sql_validator_node)
workflow.add_node("sql_executor", execute_sql_node)
workflow.add_node("result_synthesizer", result_synthesizer_node)
workflow.add_node("error_handler", handle_error_node)

workflow.set_entry_point("anonymizer")  # NEW entry
workflow.add_edge("anonymizer", "sql_generator")
workflow.add_conditional_edges(
    "sql_generator",
    route_after_generation,
    {"de_anonymizer": "de_anonymizer", "handle_error": "error_handler"},
)
workflow.add_edge("de_anonymizer", "sql_validator")
workflow.add_conditional_edges(
    "sql_validator",
    route_after_validation,
    {"sql_executor": "sql_executor", "handle_error": "error_handler"},
)
workflow.add_conditional_edges(
    "sql_executor",
    route_after_execution,
    {"result_synthesizer": "result_synthesizer", "handle_error": "error_handler"},
)
workflow.add_edge("result_synthesizer", END)
workflow.add_edge("error_handler", END)

agent_app = workflow.compile()

# -----------------------------
# 6) FLASK ROUTES
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html") if os.path.exists(os.path.join(app.template_folder or "", "index.html")) else "OK"

@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory("static", path)

def require_api_key():
    if API_KEY:
        if request.headers.get("X-API-Key") != API_KEY:
            abort(401, description="Unauthorized")

def is_sensitive_question(question: str) -> bool:
    """
    Uses a fast LLM (Groq) to classify if a user question is requesting sensitive/PII data.
    Returns True if the question should be blocked.
    """
    # Define sensitive categories (aligned with your policy)
    sensitive_categories = [
        "Social Security Number (SSN)",
        "PAN (Permanent Account Number)",
        "Aadhaar ID",
        "Passport number",
        "Credit card or debit card details",
        "CVV or card security code",
        "Phone number or mobile number",
        "Home or email address",
        "Date of birth",
        "Bank account or routing number",
        "Password or secret credential",
        "National ID or government-issued ID"
    ]
    
    prompt = f"""
You are a strict security policy enforcer for a Customer Success Management platform.
Your job is to determine if a user's question is **requesting sensitive personal information** that must be blocked.

Sensitive categories include:
{chr(10).join(f"- {cat}" for cat in sensitive_categories)}

Answer ONLY "YES" if the question is requesting any of the above.
Answer ONLY "NO" if it is not.

User question: "{question}"
Answer:
"""
    try:
        # Use your existing Groq LLM (fast and cheap)
        response = groq_llm.invoke(prompt, max_tokens=5, temperature=0.0)
        answer = response.content.strip().upper()
        return "YES" in answer
    except Exception as e:
        log.warning(f"Sensitive question classifier failed: {e}. Defaulting to SAFE.")
        return False  # fail open (or fail closed if you prefer)

@app.route("/query", methods=["POST"])
def handle_query():
    require_api_key()

    if not request.is_json:
        return jsonify({"error": "Expected JSON body"}), 400

    payload = request.get_json(silent=True) or {}
    user_question = payload.get("question")
    if is_sensitive_question(user_question):
        return jsonify({"error": "Queries about sensitive personal information are not allowed for security and compliance reasons."}), 400

    try:
        # Determine if this might be a cross-schema query by checking for multiple schema keywords
        # This is a heuristic - we'll use cross-schema info if the question mentions multiple schemas
        question_lower = user_question.lower()
       
        db_schema, filtered_hints = retrieve_context_heuristic(user_question)

        # If no relevant schema found, short-circuit and return a clear user-friendly message.
        if not db_schema:
            msg = "There is no relevant schema for this question. I can only answer questions that can be resolved using the connected database schemas."
            return jsonify({
                "sql": None,
                "final_answer": msg,
                "evals": {"reason": "no_relevant_schema"}
            })

        # Initial state
        inputs: GraphState = {
            "question": user_question.strip(),
            "db_schema": db_schema,  
            "filtered_domain_hints": filtered_hints,  
            "pii_mapping": {},
            "sanitized_schema": "",
            "anonymized_sql_query": "",
            "executable_sql_query": "",
            "error_message": "",
            "result": None,
            "final_answer": "",
        }
        # Run through full workflow
        final_state: GraphState = agent_app.invoke(inputs)

        if final_state.get("error_message"):
            return jsonify({"error": final_state["error_message"]}), 400
        
        #  Run LLM evals
        evals = run_llm_evals(user_question, db_schema, final_state)

        log.info("=== SQL Agent Evaluation Results ===")
        log.info(f"Question: {user_question}")
        log.info(f"Generated SQL: {final_state.get('executable_sql_query')}")
        log.info(f"Final Answer: {final_state.get('final_answer')}")
        log.info(f"Structural Eval: {evals.get('structural')}")
        log.info(f"Groq Eval: {evals.get('groq_eval')}")
        log.info("===================================")

        return jsonify({
            "sql": final_state.get("executable_sql_query"),
            "final_answer": final_state.get("final_answer"),   # <-- from result_synthesizer
            "evals": evals,
        })

    except Exception:
        log.exception("Unexpected server error")
        return jsonify({"error": "An internal server error occurred."}), 500


# -----------------------------
# 7) MAIN
# -----------------------------
if __name__ == "__main__":
    # In production, run with gunicorn/uvicorn. Debug only for local dev.
    app.run(debug=True, port=int(os.getenv("PORT", "5001")))
