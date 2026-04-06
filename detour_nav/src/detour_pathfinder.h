/**
 * Minimal Detour navmesh pathfinder for CARLA .bin navmeshes.
 *
 * Loads the standard Recast/Detour serialisation format (MSET header +
 * per-tile data) and exposes:
 *   - geodesic_distance(start, end) via findPath + findStraightPath
 *   - find_path(start, end) returning (distance, waypoints)
 *   - polygon flag-based obstacle blocking (setPolyFlags / setExcludeFlags)
 *   - area-type filtering (sidewalk + crosswalk only)
 */
#pragma once

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "DetourCommon.h"
#include "DetourNavMesh.h"
#include "DetourNavMeshQuery.h"

// ── CARLA-specific area types & flags ──────────────────────────────

// CARLA encodes area types as flag bits (see Navigation.h)
static constexpr unsigned short CARLA_TYPE_NONE      = 0x01;  // area 0: block
static constexpr unsigned short CARLA_TYPE_SIDEWALK   = 0x02;  // area 1
static constexpr unsigned short CARLA_TYPE_CROSSWALK  = 0x04;  // area 2
static constexpr unsigned short CARLA_TYPE_ROAD       = 0x08;  // area 3
static constexpr unsigned short CARLA_TYPE_GRASS      = 0x10;  // area 4

// Custom flag for runtime obstacle blocking (not used by CARLA itself)
static constexpr unsigned short POLY_FLAG_BLOCKED = 0x20;

// ── Navmesh file header (matches CARLA's Navigation.cpp) ──────────

#pragma pack(push, 1)
struct NavMeshSetHeader {
    int magic;       // 0x4D534554 ('MSET')
    int version;     // 1
    int numTiles;
    dtNavMeshParams params;
};

struct NavMeshTileHeader {
    unsigned int tileRef;
    int dataSize;
};
#pragma pack(pop)

static constexpr int NAVMESHSET_MAGIC   = 0x4D534554;
static constexpr int NAVMESHSET_VERSION = 1;

// ── Pathfinder ────────────────────────────────────────────────────

static constexpr int MAX_POLYS    = 512;
static constexpr int MAX_STRAIGHT = 512;

class DetourPathfinder {
public:
    DetourPathfinder()
        : navMesh_(nullptr)
        , navQuery_(nullptr)
        , loaded_(false) {
        filter_.setIncludeFlags(0xFFFF);
        filter_.setExcludeFlags(0);
    }

    ~DetourPathfinder() {
        if (navQuery_) dtFreeNavMeshQuery(navQuery_);
        if (navMesh_)  dtFreeNavMesh(navMesh_);
    }

    // Non-copyable
    DetourPathfinder(const DetourPathfinder&) = delete;
    DetourPathfinder& operator=(const DetourPathfinder&) = delete;

    // ── Loading ──

    bool load(const std::string& path) {
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp) return false;

        NavMeshSetHeader header;
        if (fread(&header, sizeof(header), 1, fp) != 1) {
            fclose(fp); return false;
        }
        if (header.magic != NAVMESHSET_MAGIC ||
            header.version != NAVMESHSET_VERSION) {
            fclose(fp); return false;
        }

        navMesh_ = dtAllocNavMesh();
        if (!navMesh_ || dtStatusFailed(navMesh_->init(&header.params))) {
            fclose(fp); return false;
        }

        for (int i = 0; i < header.numTiles; i++) {
            NavMeshTileHeader th;
            if (fread(&th, sizeof(th), 1, fp) != 1) break;
            if (th.dataSize == 0) continue;

            unsigned char* data = static_cast<unsigned char*>(
                dtAlloc(th.dataSize, DT_ALLOC_PERM));
            if (!data) { fclose(fp); return false; }

            if (fread(data, th.dataSize, 1, fp) != 1) {
                dtFree(data); fclose(fp); return false;
            }
            navMesh_->addTile(data, th.dataSize,
                              DT_TILE_FREE_DATA, th.tileRef, nullptr);
        }
        fclose(fp);

        navQuery_ = dtAllocNavMeshQuery();
        if (dtStatusFailed(navQuery_->init(navMesh_, 2048))) {
            return false;
        }

        loaded_ = true;
        return true;
    }

    bool is_loaded() const { return loaded_; }

    // ── Filter configuration ──

    void set_include_flags(unsigned short flags) {
        filter_.setIncludeFlags(flags);
    }

    void set_exclude_flags(unsigned short flags) {
        filter_.setExcludeFlags(flags);
    }

    void set_area_cost(unsigned char area, float cost) {
        filter_.setAreaCost(area, cost);
    }

    /** Restrict pathfinding to sidewalk + crosswalk polygons only. */
    void set_sidewalk_only() {
        filter_.setIncludeFlags(CARLA_TYPE_SIDEWALK | CARLA_TYPE_CROSSWALK);
        filter_.setExcludeFlags(POLY_FLAG_BLOCKED);
    }

    /** Allow all walkable area types (default). */
    void set_all_areas() {
        filter_.setIncludeFlags(0xFFFF);
        filter_.setExcludeFlags(POLY_FLAG_BLOCKED);
    }

    // ── Polygon-level obstacle blocking ──

    bool set_poly_flags(dtPolyRef ref, unsigned short flags) {
        if (!navMesh_) return false;
        return dtStatusSucceed(navMesh_->setPolyFlags(ref, flags));
    }

    bool get_poly_flags(dtPolyRef ref, unsigned short& flags) const {
        if (!navMesh_) return false;
        return dtStatusSucceed(navMesh_->getPolyFlags(ref, &flags));
    }

    /** Block all polygons whose centres fall within a 2D axis-aligned box.
     *  Returns the number of polygons blocked. */
    int block_polygons_in_aabb(float cx, float cy, float cz,
                               float half_ex, float half_ey, float half_ez) {
        if (!navMesh_ || !navQuery_) return 0;

        float center[3] = {cx, cy, cz};
        float extents[3] = {half_ex, half_ey, half_ez};

        dtPolyRef polys[MAX_POLYS];
        int npolys = 0;
        dtQueryFilter allFilter;
        allFilter.setIncludeFlags(0xFFFF);
        allFilter.setExcludeFlags(0);
        navQuery_->queryPolygons(center, extents, &allFilter,
                                 polys, &npolys, MAX_POLYS);

        int blocked = 0;
        for (int i = 0; i < npolys; i++) {
            unsigned short flags = 0;
            navMesh_->getPolyFlags(polys[i], &flags);
            if (!(flags & POLY_FLAG_BLOCKED)) {
                navMesh_->setPolyFlags(polys[i], flags | POLY_FLAG_BLOCKED);
                blockedPolys_.push_back(polys[i]);
                blocked++;
            }
        }
        return blocked;
    }

    /** Unblock all previously blocked polygons. */
    void unblock_all() {
        if (!navMesh_) return;
        for (dtPolyRef ref : blockedPolys_) {
            unsigned short flags = 0;
            navMesh_->getPolyFlags(ref, &flags);
            navMesh_->setPolyFlags(ref, flags & ~POLY_FLAG_BLOCKED);
        }
        blockedPolys_.clear();
    }

    // ── Pathfinding ──

    /** Full path query: returns (geodesic_distance, waypoints).
     *  Coordinates are in Recast frame (X-right, Y-up, Z-forward). */
    std::tuple<float, std::vector<std::array<float, 3>>>
    find_path(float sx, float sy, float sz,
              float ex, float ey, float ez) const {
        const float inf = std::numeric_limits<float>::infinity();
        if (!navMesh_ || !navQuery_)
            return {inf, {}};

        float startPos[3] = {sx, sy, sz};
        float endPos[3]   = {ex, ey, ez};
        float halfExtents[3] = {2.0f, 4.0f, 2.0f};

        dtPolyRef startRef = 0, endRef = 0;
        float nearStart[3], nearEnd[3];
        navQuery_->findNearestPoly(startPos, halfExtents, &filter_,
                                   &startRef, nearStart);
        navQuery_->findNearestPoly(endPos, halfExtents, &filter_,
                                   &endRef, nearEnd);
        if (!startRef || !endRef)
            return {inf, {}};

        // A* on polygon graph
        dtPolyRef polys[MAX_POLYS];
        int npolys = 0;
        navQuery_->findPath(startRef, endRef, nearStart, nearEnd,
                            &filter_, polys, &npolys, MAX_POLYS);
        if (npolys == 0)
            return {inf, {}};

        // Snap end position to last polygon if path is partial
        float actualEnd[3];
        dtVcopy(actualEnd, nearEnd);
        if (polys[npolys - 1] != endRef) {
            navQuery_->closestPointOnPoly(
                polys[npolys - 1], nearEnd, actualEnd, nullptr);
        }

        // Funnel algorithm -> straight path waypoints
        float straightPath[MAX_STRAIGHT * 3];
        int nstraight = 0;
        navQuery_->findStraightPath(nearStart, actualEnd,
                                     polys, npolys,
                                     straightPath, nullptr, nullptr,
                                     &nstraight, MAX_STRAIGHT);
        if (nstraight == 0)
            return {inf, {}};

        // Sum segment lengths = geodesic distance
        float dist = 0.0f;
        std::vector<std::array<float, 3>> points(nstraight);
        for (int i = 0; i < nstraight; i++) {
            points[i] = {
                straightPath[i * 3],
                straightPath[i * 3 + 1],
                straightPath[i * 3 + 2],
            };
            if (i > 0) {
                float dx = straightPath[i * 3]     - straightPath[(i-1) * 3];
                float dy = straightPath[i * 3 + 1] - straightPath[(i-1) * 3 + 1];
                float dz = straightPath[i * 3 + 2] - straightPath[(i-1) * 3 + 2];
                dist += std::sqrt(dx*dx + dy*dy + dz*dz);
            }
        }
        return {dist, points};
    }

    /** Convenience: geodesic distance only, no path allocation. */
    float geodesic_distance(float sx, float sy, float sz,
                            float ex, float ey, float ez) const {
        auto [dist, _] = find_path(sx, sy, sz, ex, ey, ez);
        return dist;
    }

    /** Random navigable point on the navmesh (wraps dtNavMeshQuery). */
    std::array<float, 3> get_random_point() const {
        if (!navQuery_) return {0, 0, 0};
        dtPolyRef ref = 0;
        float pt[3] = {0, 0, 0};
        // dtNavMeshQuery needs a random function
        auto randFunc = []() -> float {
            return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        };
        navQuery_->findRandomPoint(&filter_, randFunc, &ref, pt);
        return {pt[0], pt[1], pt[2]};
    }

private:
    dtNavMesh*              navMesh_;
    dtNavMeshQuery*         navQuery_;
    dtQueryFilter           filter_;
    bool                    loaded_;
    std::vector<dtPolyRef>  blockedPolys_;   // tracked for unblock_all()
};
