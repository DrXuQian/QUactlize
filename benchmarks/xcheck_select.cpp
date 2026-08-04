// DO THE TWO SELECTION PROCEDURES AGREE? -- the precondition for deleting the C++ one.
//
// benchmarks/bench_select.hpp decides inside the bench; benchmarks/analyse.py decides outside it. The plan
// (docs/BENCH_DESIGN.md step 3) is to delete the first once the second reproduces it. "Reproduces it" has to be
// a check, because the alternative is replacing one unverified decision procedure with another -- and the whole
// argument for moving selection out is that a procedure nobody can feed planted data to is a procedure nobody
// has tested.
//
// So this program runs the C++ code over a sample file and prints its verdict in a form the test can diff
// against `analyse.py --json`. It parses only the fields it needs, by hand: pulling in a JSON library for a
// cross-check would make the two paths differ by their parsers as well as their logic.
//
//   c++ -std=c++17 -Ibenchmarks benchmarks/xcheck_select.cpp -o /tmp/xcheck && /tmp/xcheck run.jsonl
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "bench_select.hpp"

namespace {

// Minimal field readers. `"key":value` with no nesting and no escapes -- which is exactly what
// bench_samples.hpp writes, and it refuses to write anything else (it rejects a quote in a name field).
bool str_field(std::string const& l, char const* key, std::string& out) {
  const std::string pat = std::string("\"") + key + "\":\"";
  const size_t a = l.find(pat);
  if (a == std::string::npos) return false;
  const size_t b = l.find('"', a + pat.size());
  if (b == std::string::npos) return false;
  out = l.substr(a + pat.size(), b - a - pat.size());
  return true;
}

bool num_field(std::string const& l, char const* key, double& out) {
  const std::string pat = std::string("\"") + key + "\":";
  const size_t a = l.find(pat);
  if (a == std::string::npos) return false;
  out = std::strtod(l.c_str() + a + pat.size(), nullptr);
  return true;
}

std::string config_name(std::string const& l) {
  std::string schema;
  double tm, tn, tk, wm, wn, st;
  if (!str_field(l, "schema", schema) || !num_field(l, "tm", tm) || !num_field(l, "tn", tn) ||
      !num_field(l, "tk", tk) || !num_field(l, "wm", wm) || !num_field(l, "wn", wn) ||
      !num_field(l, "st", st))
    return "";
  char buf[96];
  std::snprintf(buf, sizeof buf, "%s %dx%dx%d:%dx%d:s%d", schema.c_str(), int(tm), int(tn), int(tk),
                int(wm), int(wn), int(st));
  return buf;
}

// The fixture key must match analyse.py's FIXTURE_KEYS exactly, or the two group differently and the diff
// reports a disagreement that is really a schema mismatch.
std::string fixture_key(std::string const& l) {
  std::string fx, dist;
  double n, k, gs, ex, rows, mmax;
  if (!str_field(l, "fixture", fx) || !str_field(l, "dist", dist) || !num_field(l, "n", n) ||
      !num_field(l, "k", k) || !num_field(l, "gs", gs) || !num_field(l, "experts", ex) ||
      !num_field(l, "rows", rows) || !num_field(l, "mmax", mmax))
    return "";
  char buf[256];
  std::snprintf(buf, sizeof buf, "%s|%s|%d|%d|%d|%d|%d|%d", fx.c_str(), dist.c_str(), int(n), int(k),
                int(gs), int(ex), int(rows), int(mmax));
  return buf;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: xcheck_select <run.jsonl>\n"); return 2; }
  std::FILE* f = std::fopen(argv[1], "r");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }

  std::map<std::string, Best> per_fixture;
  std::map<std::string, int> passes;
  char line[2048];
  while (std::fgets(line, sizeof line, f)) {
    std::string l(line);
    if (l.find("\"rec\":\"s\"") == std::string::npos) continue;
    const std::string fk = fixture_key(l), cn = config_name(l);
    double us, pass;
    if (fk.empty() || cn.empty() || !num_field(l, "us", us) || !num_field(l, "pass", pass)) continue;
    upd(per_fixture[fk], cn.c_str(), us);
    int& p = passes[fk];
    if (int(pass) + 1 > p) p = int(pass) + 1;
  }
  std::fclose(f);

  // Sorted output so a diff is stable. One block per fixture, mirroring analyse.py's verdict fields.
  for (auto& [fk, b] : per_fixture) {
    const int ties = settle(b);
    std::printf("fixture %s\n", fk.c_str());
    std::printf("  passes %d\n", passes[fk]);
    std::printf("  leader %s\n", b.tag);
    std::printf("  median %.6f\n", b.us);
    double lo = 0, hi = 0;
    for (auto const& s : b.seen)
      if (std::strncmp(s.tag, b.tag, 64) == 0) {
        lo = *std::min_element(s.us.begin(), s.us.end());
        hi = *std::max_element(s.us.begin(), s.us.end());
      }
    std::printf("  band %.6f %.6f\n", lo, hi);
    std::printf("  ties %d\n", ties);
    std::vector<std::string> names;
    for (auto const& s : b.seen) {
      if (std::strncmp(s.tag, b.tag, 64) == 0) continue;
      if (*std::min_element(s.us.begin(), s.us.end()) <= hi) names.push_back(s.tag);
    }
    std::sort(names.begin(), names.end());
    for (auto const& n : names) std::printf("  tie %s\n", n.c_str());
  }
  return 0;
}
