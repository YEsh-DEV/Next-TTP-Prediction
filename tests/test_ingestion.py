import os
import sys
import pytest
from neo4j import GraphDatabase
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

@pytest.fixture(scope="module")
def neo4j_session():
    if not URI or not USERNAME or not PASSWORD:
        pytest.skip("Neo4j credentials not set. Cannot run graph tests.")
        
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    session = driver.session()
    yield session
    session.close()
    driver.close()

def test_apt_group_exists(neo4j_session):
    result = neo4j_session.run("MATCH (g:APTGroup {name: 'APT30'}) RETURN g")
    record = result.single()
    assert record is not None, "APT30 should exist in the graph"

def test_technique_exists(neo4j_session):
    result = neo4j_session.run("MATCH (t:Technique {id: 'T1059'}) RETURN t")
    record = result.single()
    assert record is not None, "T1059 should exist in the graph"

def test_software_exists(neo4j_session):
    result = neo4j_session.run("MATCH (s:Software {id: 'S0031'}) RETURN s")
    record = result.single()
    assert record is not None, "S0031 should exist in the graph"

def test_relationship_exists(neo4j_session):
    # Test if APT30 uses S0031
    result = neo4j_session.run("MATCH (g:APTGroup {name: 'APT30'})-[:USES]->(s:Software {id: 'S0031'}) RETURN s")
    assert result.single() is not None, "APT30 should use Software S0031"

def test_technique_belongs_to_tactic(neo4j_session):
    # Test if T1059 belongs to a Tactic
    result = neo4j_session.run("MATCH (t:Technique {id: 'T1059'})-[:BELONGS_TO_TACTIC]->(ta:Tactic) RETURN ta")
    assert result.single() is not None, "T1059 should belong to a Tactic"

def test_software_uses_technique(neo4j_session):
    # Test if S0031 uses technique T1059
    result = neo4j_session.run("MATCH (s:Software {id: 'S0031'})-[:USES_TECHNIQUE]->(t:Technique {id: 'T1059'}) RETURN t")
    assert result.single() is not None, "S0031 should use Technique T1059"
