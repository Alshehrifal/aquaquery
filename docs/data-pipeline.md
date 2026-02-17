# Data Pipeline

## Overview

AquaQuery accesses Argo oceanographic data through the argopy library, which
handles fetching from the Global Data Assembly Centre (GDAC). Data is cached
locally for performance.

## argopy Fetch Strategy

### Data Access Mode (ArgoDataManager)
- **Primary:** argopy DataFetcher with `src='erddap'` (fast, reliable)
- **Fallback:** argopy with `src='gdac'` (official source, slower)
- **Cache:** File-based NetCDF caching at `data/cache/` (MD5-keyed .nc files)

### Legacy Mode (ArgoDataLoader -- used for metadata only)
- **Primary:** argopy DataFetcher with `src='gdac'`
- **Fallback:** argopy with `src='erddap'`
- **Cache:** argopy built-in caching at `data/sample/`

### Fetch Patterns
```python
# By region (most common for AquaQuery)
fetcher = DataFetcher(src='gdac').region(
    [-80, 0, -60, 60, 0, 2000, '2020-01', '2024-12']
    # [lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, date_min, date_max]
)

# By float ID
fetcher = DataFetcher(src='gdac').float(6902746)

# By profile
fetcher = DataFetcher(src='gdac').profile(6902746, 1)
```

### PoC Demo Region
For fast demos, pre-fetch a small region:
- North Atlantic: lat 20-50, lon -60 to -20
- Depth: 0-2000m
- Time: 2023-2024
- ~500-1000 profiles, manageable size

## Caching Strategy

### ArgoDataManager File Cache (primary)
Query parameters are hashed to an MD5 key. Cached results are stored as `.nc` files:
```
data/cache/<md5_hash>.nc
```

- **Cache hit:** Load from disk (<1s)
- **Cache miss:** Fetch from ERDDAP (~10-20s), then GDAC fallback (~30-60s)
- **Cache invalidation:** Manual (delete files in `data/cache/`)
- **Key components:** lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, start_date, end_date

### argopy Built-in Cache (legacy)
```python
argopy.set_options(cachedir='data/sample/', cache=True)
```
Used by `ArgoDataLoader` for metadata-only operations.

## NetCDF Data Structure

Argo profiles in xarray Dataset format:

### Dimensions
- `N_PROF`: Number of profiles
- `N_LEVELS`: Depth levels per profile

### Key Variables
| Variable | Unit | Description |
|----------|------|-------------|
| `TEMP` | degC | In-situ temperature |
| `PSAL` | PSU | Practical salinity |
| `PRES` | dbar | Sea pressure (~depth in meters) |
| `DOXY` | umol/kg | Dissolved oxygen |
| `LATITUDE` | degrees_north | Profile latitude |
| `LONGITUDE` | degrees_east | Profile longitude |
| `JULD` | days since 1950-01-01 | Julian date |

### Quality Control Flags
Each variable has a companion QC variable (e.g., `TEMP_QC`):
- 1: Good data
- 2: Probably good
- 3: Probably bad (exclude)
- 4: Bad data (exclude)
- 9: Missing value

AquaQuery keeps only QC flags 1 and 2.

## Embedding Content

### Documents for ChromaDB Vector Store

~50 documents covering:

1. **Argo Program** (5 docs)
   - What is Argo? Global array of 4000+ profiling floats
   - How floats work (dive cycle, parking depth, profiling)
   - History and goals of the program
   - Coverage (global ocean, except ice-covered regions)
   - Data availability and access

2. **Variables** (8 docs)
   - Temperature: what it measures, typical ranges, significance
   - Salinity: practical salinity units, ocean patterns
   - Pressure: relationship to depth, units
   - Dissolved oxygen: biological significance, oxygen minimum zones
   - Quality control: flag meanings, data reliability
   - Mixed layer depth: definition, seasonal cycle
   - Thermocline/halocline: structure, importance
   - Water masses: identification by T-S properties

3. **Ocean Basins** (6 docs)
   - Pacific Ocean: largest, deepest, characteristics
   - Atlantic Ocean: meridional overturning circulation
   - Indian Ocean: monsoon influence
   - Southern Ocean: Antarctic circumpolar current
   - Arctic Ocean: ice cover, limited Argo coverage
   - Mediterranean Sea: high salinity basin

4. **Data Concepts** (6 docs)
   - Profile: vertical slice of ocean at one location/time
   - Climatology: long-term average conditions
   - Anomaly: deviation from climatology
   - Mixed layer: well-mixed surface layer
   - Deep water formation: dense water sinking
   - El Nino/La Nina: Pacific temperature patterns

### Embedding Model
- **sentence-transformers/all-MiniLM-L6-v2**
- 384-dimensional vectors
- Fast inference (~100ms per document)
- No API key required (runs locally)

## Data Flow Summary

```
User Query -> Query Agent -> ArgoDataManager -> File Cache (.nc)
                                                    |
                                              (cache miss)
                                                    |
                                            ERDDAP -> GDAC (fallback)
                                                    |
                                              xarray Dataset
                                                    |
                                            QC Filter (flags 1,2)
                                                    |
                                            Save to cache (.nc)
                                                    |
                                            Aggregation/Stats
                                                    |
                                            Structured Response
```
