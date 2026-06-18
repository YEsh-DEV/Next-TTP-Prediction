import os
import sys
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Ensure the root directory is in the sys.path to find the .env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

def run_query(query):
    if not URI or not USERNAME or not PASSWORD:
        print("Error: Missing Neo4j credentials in .env file.")
        sys.exit(1)
        
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    try:
        with driver.session() as session:
            result = session.run(query)
            # Fetch all records and their data
            records = [record.data() for record in result]
            
            if not records:
                print("Query executed successfully. (0 records returned)")
            else:
                print(f"Returned {len(records)} records:")
                for idx, record in enumerate(records):
                    print(f"\n--- Record {idx + 1} ---")
                    for key, value in record.items():
                        print(f"{key}: {value}")
                        
    except Exception as e:
        print(f"An error occurred executing the query:\n{e}")
    finally:
        driver.close()

if __name__ == "__main__":
    # If the user provides a query directly as an argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_query(query)
    else:
        # Otherwise, start an interactive loop
        print("--- Neo4j Interactive CLI ---")
        print("Type your Cypher query and press Enter.")
        print("Type 'exit' or 'quit' to close.")
        while True:
            try:
                query = input("\ncypher> ")
                if query.strip().lower() in ['exit', 'quit']:
                    break
                if not query.strip():
                    continue
                run_query(query)
            except KeyboardInterrupt:
                break
            except EOFError:
                break
