"""ChromaDB vector store indexer for Argo knowledge base."""

import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from backend.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Knowledge base documents about Argo and oceanography
ARGO_KNOWLEDGE_DOCS: list[dict[str, str]] = [
    # Argo Program
    {
        "id": "argo_overview",
        "category": "argo_program",
        "content": (
            "Argo is a global array of over 4,000 free-drifting profiling floats that measure "
            "temperature, salinity, and increasingly oxygen and other biogeochemical properties "
            "of the upper 2,000 meters of the ocean. Argo is the largest source of subsurface "
            "ocean observations ever collected. The data are freely available in near real-time."
        ),
    },
    {
        "id": "argo_floats",
        "category": "argo_program",
        "content": (
            "Argo floats are autonomous instruments that drift with ocean currents at a parking "
            "depth (typically 1,000m). Every 10 days, a float descends to 2,000m then ascends "
            "to the surface, measuring temperature and salinity along the way. At the surface, "
            "data is transmitted via satellite before the float descends again. Each float "
            "operates for 4-5 years on battery power."
        ),
    },
    {
        "id": "argo_history",
        "category": "argo_program",
        "content": (
            "The Argo program began in 1999 as an international collaboration. It reached its "
            "target of 3,000 floats in 2007 and has since expanded. Over 30 countries contribute "
            "to deploying and maintaining the array. Argo is a key component of the Global Ocean "
            "Observing System (GOOS) and provides critical data for climate research, weather "
            "forecasting, and ocean state estimation."
        ),
    },
    {
        "id": "argo_coverage",
        "category": "argo_program",
        "content": (
            "Argo floats cover most of the global ocean between 60S and 60N latitude. Coverage "
            "is sparser in ice-covered polar regions, marginal seas, and near coasts. The array "
            "provides approximately one profile per 3-degree square every 10 days. Deep Argo "
            "floats extend measurements below 2,000m to the ocean floor (up to 6,000m)."
        ),
    },
    {
        "id": "argo_data_access",
        "category": "argo_program",
        "content": (
            "Argo data is freely available from two Global Data Assembly Centres (GDACs): "
            "one in France (Coriolis) and one in the US (GODAE). Data is available in NetCDF "
            "format and can be accessed via FTP, HTTP, or through tools like argopy (Python), "
            "argodata (R), or the Argo data selection tool. Real-time data is available within "
            "24 hours; delayed-mode quality-controlled data within 12 months."
        ),
    },
    # Variables
    {
        "id": "var_temperature",
        "category": "variables",
        "content": (
            "Ocean temperature (TEMP) is measured in degrees Celsius. Surface temperatures "
            "range from about -2 degC in polar regions to 30+ degC in tropical waters. Temperature "
            "generally decreases with depth, with the steepest gradient in the thermocline "
            "(typically 200-1000m depth). Deep ocean temperatures are typically 1-4 degC. "
            "Temperature is crucial for understanding ocean heat content, circulation, and "
            "climate change."
        ),
    },
    {
        "id": "var_salinity",
        "category": "variables",
        "content": (
            "Practical salinity (PSAL) is measured in PSU (Practical Salinity Units). Open "
            "ocean salinity typically ranges from 33 to 37 PSU. Higher values occur in "
            "evaporation-dominated regions (e.g., Mediterranean Sea, ~38-39 PSU) and lower "
            "values near river mouths and in regions of heavy rainfall. Salinity affects water "
            "density and is key to identifying water masses and understanding ocean circulation."
        ),
    },
    {
        "id": "var_pressure",
        "category": "variables",
        "content": (
            "Sea pressure (PRES) is measured in decibars (dbar), which approximately equals "
            "depth in meters. Standard Argo floats profile from 0 to 2,000 dbar. Pressure "
            "increases roughly linearly with depth at about 1 dbar per meter. Pressure is "
            "used as the vertical coordinate for ocean profiles."
        ),
    },
    {
        "id": "var_oxygen",
        "category": "variables",
        "content": (
            "Dissolved oxygen (DOXY) is measured in micromoles per kilogram (umol/kg). Surface "
            "oxygen is typically near saturation (200-300 umol/kg). Oxygen minimum zones (OMZs) "
            "at intermediate depths (200-1000m) can have values below 20 umol/kg, particularly "
            "in the Eastern Pacific and Northern Indian Ocean. About 30% of Argo floats now "
            "carry oxygen sensors as part of the Biogeochemical-Argo program."
        ),
    },
    {
        "id": "var_qc",
        "category": "variables",
        "content": (
            "Argo quality control (QC) flags indicate data reliability. Flag 1 means good "
            "data, flag 2 means probably good, flag 3 means probably bad, flag 4 means bad "
            "data, and flag 9 indicates missing values. Real-time QC applies automatic tests; "
            "delayed-mode QC involves expert review and statistical comparison with climatology. "
            "For reliable analysis, use only data with QC flags 1 or 2."
        ),
    },
    # Ocean concepts
    {
        "id": "concept_thermocline",
        "category": "ocean_concepts",
        "content": (
            "The thermocline is a layer of water where temperature changes rapidly with depth. "
            "In the ocean, the permanent thermocline exists between roughly 200m and 1,000m "
            "depth, separating warm surface waters from cold deep waters. A seasonal thermocline "
            "develops in summer in the upper 50-200m and erodes in winter due to surface cooling "
            "and wind mixing. The thermocline acts as a barrier to vertical mixing."
        ),
    },
    {
        "id": "concept_halocline",
        "category": "ocean_concepts",
        "content": (
            "The halocline is a layer where salinity changes rapidly with depth. It is most "
            "prominent in polar regions and near river outflows where fresh water overlies "
            "saltier water. In the Arctic Ocean, the halocline is a critical feature that "
            "insulates sea ice from warmer Atlantic water below. In tropical/subtropical "
            "oceans, the halocline is typically weaker than the thermocline."
        ),
    },
    {
        "id": "concept_mixed_layer",
        "category": "ocean_concepts",
        "content": (
            "The mixed layer is the near-surface layer of the ocean where temperature, "
            "salinity, and density are nearly uniform due to wind-driven turbulent mixing. "
            "Mixed layer depth varies from less than 20m in calm tropical waters to over "
            "500m in winter storm regions. It shoals in summer (due to surface warming) "
            "and deepens in winter (due to cooling and storms). The mixed layer depth is "
            "important for air-sea interaction and biological productivity."
        ),
    },
    {
        "id": "concept_water_masses",
        "category": "ocean_concepts",
        "content": (
            "Water masses are large bodies of water with distinct temperature-salinity (T-S) "
            "characteristics formed at the ocean surface. Examples include Antarctic Bottom "
            "Water (very cold, moderate salinity), North Atlantic Deep Water (cold, high salinity), "
            "and Mediterranean Water (warm, very high salinity). Water masses can be identified "
            "on T-S diagrams and tracked as they spread through the ocean interior."
        ),
    },
    # Ocean basins
    {
        "id": "basin_pacific",
        "category": "ocean_basins",
        "content": (
            "The Pacific Ocean is the largest and deepest ocean basin, covering about "
            "165 million square kilometers. It contains the deepest point on Earth (Mariana "
            "Trench, ~11,000m). The Pacific features the El Nino-Southern Oscillation (ENSO), "
            "the most important mode of climate variability. The western Pacific warm pool "
            "has the warmest surface temperatures globally (>28 degC)."
        ),
    },
    {
        "id": "basin_atlantic",
        "category": "ocean_basins",
        "content": (
            "The Atlantic Ocean is the second-largest ocean, known for the Atlantic Meridional "
            "Overturning Circulation (AMOC) - a major system of currents including the Gulf "
            "Stream. The AMOC transports warm water northward and cold deep water southward. "
            "The Atlantic has higher average salinity than other oceans due to net evaporation "
            "and limited connection to lower-salinity Pacific waters."
        ),
    },
    {
        "id": "basin_indian",
        "category": "ocean_basins",
        "content": (
            "The Indian Ocean is the third-largest ocean, strongly influenced by the Asian "
            "monsoon system. Seasonal reversal of winds drives large changes in surface "
            "currents, temperature, and biological productivity. The Arabian Sea experiences "
            "intense upwelling and one of the world's most prominent oxygen minimum zones."
        ),
    },
    {
        "id": "basin_southern",
        "category": "ocean_basins",
        "content": (
            "The Southern Ocean surrounds Antarctica and is home to the Antarctic Circumpolar "
            "Current (ACC), the largest ocean current by volume transport. It is a critical "
            "region for global ocean circulation, CO2 uptake, and Antarctic Bottom Water "
            "formation. Argo coverage is improving but remains challenging due to sea ice."
        ),
    },
    # Data concepts
    {
        "id": "concept_profile",
        "category": "data_concepts",
        "content": (
            "An ocean profile is a vertical measurement of water properties (temperature, "
            "salinity, etc.) at a single location and time. Argo floats measure profiles "
            "from 2,000m depth to the surface every 10 days. A profile typically contains "
            "measurements at 70-100 depth levels. Profiles are the fundamental unit of "
            "Argo data and can be analyzed individually or aggregated for regional studies."
        ),
    },
    {
        "id": "concept_climatology",
        "category": "data_concepts",
        "content": (
            "Ocean climatology refers to long-term average conditions, typically computed "
            "over 30+ years. The World Ocean Atlas (WOA) is a widely used climatology product. "
            "Anomalies are deviations from climatological averages and are used to identify "
            "unusual conditions. Argo data is now long enough (25+ years) to contribute "
            "significantly to modern ocean climatologies."
        ),
    },
    {
        "id": "concept_enso",
        "category": "data_concepts",
        "content": (
            "El Nino and La Nina are phases of the El Nino-Southern Oscillation (ENSO), "
            "characterized by anomalous warming (El Nino) or cooling (La Nina) of the "
            "eastern equatorial Pacific. ENSO affects global weather patterns, marine "
            "ecosystems, and ocean heat distribution. Argo floats are crucial for monitoring "
            "subsurface temperature changes that precede ENSO events."
        ),
    },
]


class ArgoKnowledgeIndexer:
    """Manages the ChromaDB vector store for Argo knowledge base."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._client = chromadb.PersistentClient(
            path=str(self._settings.embeddings_dir)
        )
        self._collection: chromadb.Collection | None = None

    @property
    def collection(self) -> chromadb.Collection:
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def index_knowledge_base(self) -> int:
        """Index all Argo knowledge documents into ChromaDB.

        Returns the number of documents indexed.
        """
        existing = self.collection.count()
        if existing >= len(ARGO_KNOWLEDGE_DOCS):
            logger.info(
                "Knowledge base already indexed (%d docs), skipping", existing
            )
            return existing

        ids = [doc["id"] for doc in ARGO_KNOWLEDGE_DOCS]
        documents = [doc["content"] for doc in ARGO_KNOWLEDGE_DOCS]
        metadatas = [{"category": doc["category"]} for doc in ARGO_KNOWLEDGE_DOCS]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        count = self.collection.count()
        logger.info("Indexed %d documents into ChromaDB", count)
        return count

    def search(
        self,
        query: str,
        top_k: int | None = None,
        category: str | None = None,
    ) -> list[dict[str, str]]:
        """Search the knowledge base for relevant documents.

        Args:
            query: Natural language search query
            top_k: Number of results to return (default from settings)
            category: Optional category filter

        Returns:
            List of dicts with 'id', 'content', 'category', 'distance' keys
        """
        k = top_k or self._settings.rag_top_k

        where_filter = {"category": category} if category else None

        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where_filter,
        )

        documents = []
        for i in range(len(results["ids"][0])):
            documents.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "category": results["metadatas"][0][i].get("category", ""),
                "distance": str(results["distances"][0][i]),
            })

        return documents

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self._client.delete_collection(self._settings.chroma_collection_name)
        self._collection = None
        logger.info("Reset ChromaDB collection")
