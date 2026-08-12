// Durable raw JSONL records for benchmarks/sweep_gemv_perf.py.
//
// This deliberately does not reuse bench_samples.hpp: GEMV tactics have a
// different identity, and silently dropping a newly added tactic axis from a
// generic key would merge distinct kernels.  Complete shape/config JSON objects
// are repeated on every record and checked again by the Python reader.
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gemv_perf_timing.hpp"

namespace gemv_perf_samples {

inline constexpr char kRawSchema[] = "gemv-sweep-raw-v1";

struct Run {
  std::string run_id;
  std::string build;
  std::string space_id;
  bool partial_space = false;
};

struct Candidate {
  std::string run_id;
  std::string shape_id;
  std::string shape_json;   // complete JSON object, not an ID-derived lookup
  std::string format;
  std::string config_id;
  std::string config_json;  // complete JSON object, including every tactic axis
};

struct Attempt {
  Candidate candidate;
  std::string attempt_id;
  std::uint32_t pass = 0;
};

namespace detail {

inline void append_json_string(std::string& out, std::string const& value) {
  static char const hex[] = "0123456789abcdef";
  out.push_back('"');
  for (unsigned char c : value) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          out += "\\u00";
          out.push_back(hex[c >> 4]);
          out.push_back(hex[c & 0x0f]);
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out.push_back('"');
}

inline bool json_object_text(std::string const& text) {
  if (text.empty() || text.front() != '{' || text.back() != '}') return false;
  return text.find('\n') == std::string::npos && text.find('\r') == std::string::npos;
}

inline std::string attempt_key(Attempt const& attempt) {
  // Length-prefix every field; separators can therefore occur in user strings
  // without aliasing two identities inside the writer's state machine.
  std::string out;
  auto add = [&](std::string const& field) {
    out += std::to_string(field.size());
    out.push_back(':');
    out += field;
  };
  add(attempt.candidate.run_id);
  add(attempt.candidate.shape_id);
  add(attempt.candidate.shape_json);
  add(attempt.candidate.format);
  add(attempt.candidate.config_id);
  add(attempt.candidate.config_json);
  add(attempt.attempt_id);
  out += "#" + std::to_string(attempt.pass);
  return out;
}

inline void append_candidate(std::string& out, Attempt const& attempt) {
  auto field = [&](char const* name, std::string const& value) {
    out += ",\"";
    out += name;
    out += "\":";
    append_json_string(out, value);
  };
  field("run_id", attempt.candidate.run_id);
  field("shape_id", attempt.candidate.shape_id);
  out += ",\"shape\":" + attempt.candidate.shape_json;
  field("format", attempt.candidate.format);
  field("config_id", attempt.candidate.config_id);
  out += ",\"config\":" + attempt.candidate.config_json;
  field("attempt_id", attempt.attempt_id);
  out += ",\"pass\":" + std::to_string(attempt.pass);
}

}  // namespace detail

class JsonlWriter {
 public:
  // If path is null, GEMV_SWEEP_JSONL is used.  An unset variable disables
  // writing without changing benchmark behaviour.  A set but unopenable path
  // is a hard writer failure and every write method returns false.
  explicit JsonlWriter(char const* path = nullptr) {
    char const* selected = path ? path : std::getenv("GEMV_SWEEP_JSONL");
    if (!selected || !*selected) return;
    requested_ = true;
    file_ = std::fopen(selected, "a");
    if (!file_) {
      fail(std::string("cannot append GEMV sweep JSONL ") + selected + ": " +
           std::strerror(errno));
    }
  }

  ~JsonlWriter() {
    if (file_) std::fclose(file_);
  }

  JsonlWriter(JsonlWriter const&) = delete;
  JsonlWriter& operator=(JsonlWriter const&) = delete;

  bool requested() const { return requested_; }
  bool enabled() const { return file_ != nullptr && error_.empty(); }
  bool ok() const { return error_.empty(); }
  std::string const& error() const { return error_; }

  bool write_run(Run const& run) {
    if (!requested_) return true;
    if (!ready()) return false;
    if (have_run_) return fail("duplicate run header in one writer");
    if (run.run_id.empty() || run.build.empty() || run.space_id.empty())
      return fail("run_id/build/space_id must be non-empty");
    run_id_ = run.run_id;
    std::string line = "{\"rec\":\"run\",\"schema\":\"";
    line += kRawSchema;
    line += "\",\"run_id\":";
    detail::append_json_string(line, run.run_id);
    line += ",\"build\":";
    detail::append_json_string(line, run.build);
    line += ",\"space_id\":";
    detail::append_json_string(line, run.space_id);
    line += ",\"partial_space\":";
    line += run.partial_space ? "true}" : "false}";
    if (!write_line(line)) return false;
    have_run_ = true;
    return true;
  }

  bool write_attempt(Attempt const& attempt, std::uint32_t expected_samples) {
    if (!requested_) return true;
    if (!validate_attempt(attempt) || expected_samples == 0)
      return expected_samples == 0 ? fail("expected_samples must be positive") : false;
    std::string const key = detail::attempt_key(attempt);
    if (states_.count(key)) return fail("duplicate attempt record");
    State state;
    state.expected = expected_samples;
    state.seen.assign(expected_samples, false);
    states_.emplace(key, std::move(state));

    std::string line = record_prefix("attempt");
    detail::append_candidate(line, attempt);
    line += ",\"expected_samples\":" + std::to_string(expected_samples) + "}";
    if (!write_line(line)) {
      states_.erase(key);
      return false;
    }
    return true;
  }

  bool write_sample(
      Attempt const& attempt,
      std::uint32_t launch_index,
      gemv_perf_timing::RawEventSample const& sample) {
    if (!requested_) return true;
    if (!validate_attempt(attempt)) return false;
    std::string const key = detail::attempt_key(attempt);
    auto it = states_.find(key);
    if (it == states_.end()) return fail("sample has no preceding attempt");
    State& state = it->second;
    if (state.excluded) return fail("sample follows an exclusion");
    if (launch_index >= state.expected) return fail("sample launch_index exceeds expected_samples");
    if (state.seen[launch_index]) return fail("duplicate sample launch_index");
    if (!std::isfinite(sample.event_ms) || sample.event_ms <= 0.0f ||
        !std::isfinite(sample.event_us) || sample.event_us <= 0.0)
      return fail("sample is non-finite or non-positive");
    std::uint32_t bits = 0;
    std::memcpy(&bits, &sample.event_ms, sizeof(bits));
    if (bits != sample.event_ms_bits)
      return fail("sample event_ms_bits disagrees with event_ms");
    double const exact_us = static_cast<double>(sample.event_ms) * 1000.0;
    if (std::fabs(exact_us - sample.event_us) > 1.0e-6)
      return fail("sample event_us disagrees with raw event_ms");

    char event_us[64];
    int const n = std::snprintf(event_us, sizeof(event_us), "%.9f", exact_us);
    if (n <= 0 || static_cast<std::size_t>(n) >= sizeof(event_us))
      return fail("cannot format event_us");
    std::string line = record_prefix("sample");
    detail::append_candidate(line, attempt);
    line += ",\"launch_index\":" + std::to_string(launch_index);
    line += ",\"event_ms_bits\":" + std::to_string(bits);
    line += ",\"event_us\":";
    line += event_us;
    line += "}";
    if (!write_line(line)) return false;
    state.seen[launch_index] = true;
    return true;
  }

  bool write_samples(Attempt const& attempt, gemv_perf_timing::RawEventBatch const& batch) {
    if (!requested_) return true;
    if (!batch.complete()) return fail("refusing to write an incomplete raw event batch");
    for (std::size_t i = 0; i < batch.samples.size(); ++i) {
      if (!write_sample(attempt, static_cast<std::uint32_t>(i), batch.samples[i])) return false;
    }
    return true;
  }

  bool write_excluded(Attempt const& attempt, std::string const& why) {
    if (!requested_) return true;
    if (!validate_attempt(attempt) || why.empty())
      return why.empty() ? fail("exclusion reason must be non-empty") : false;
    std::string const key = detail::attempt_key(attempt);
    auto it = states_.find(key);
    if (it == states_.end()) return fail("exclusion has no preceding attempt");
    State& state = it->second;
    if (state.excluded) return fail("duplicate exclusion record");
    for (bool seen : state.seen)
      if (seen) return fail("an attempt cannot have samples and an exclusion");

    std::string line = record_prefix("excluded");
    detail::append_candidate(line, attempt);
    line += ",\"why\":";
    detail::append_json_string(line, why);
    line += "}";
    if (!write_line(line)) return false;
    state.excluded = true;
    return true;
  }

 private:
  struct State {
    std::uint32_t expected = 0;
    std::vector<bool> seen;
    bool excluded = false;
  };

  bool ready() {
    if (!requested_) return false;
    if (!error_.empty()) return false;
    if (!file_) return fail("JSONL stream is unavailable");
    return true;
  }

  bool validate_attempt(Attempt const& attempt) {
    if (!ready()) return false;
    if (!have_run_) return fail("candidate record precedes run header");
    if (attempt.candidate.run_id != run_id_) return fail("candidate run_id differs from run header");
    if (attempt.candidate.shape_id.empty() || attempt.candidate.format.empty() ||
        attempt.candidate.config_id.empty() || attempt.attempt_id.empty())
      return fail("shape_id/format/config_id/attempt_id must be non-empty");
    if (!detail::json_object_text(attempt.candidate.shape_json) ||
        !detail::json_object_text(attempt.candidate.config_json))
      return fail("shape/config must be one-line JSON objects");
    return true;
  }

  std::string record_prefix(char const* rec) const {
    std::string out = "{\"rec\":\"";
    out += rec;
    out += "\",\"schema\":\"";
    out += kRawSchema;
    out += "\"";
    return out;
  }

  bool write_line(std::string const& line) {
    if (!ready()) return false;
    if (std::fwrite(line.data(), 1, line.size(), file_) != line.size() ||
        std::fputc('\n', file_) == EOF || std::fflush(file_) != 0) {
      return fail(std::string("JSONL append/flush failed: ") + std::strerror(errno));
    }
    return true;
  }

  bool fail(std::string message) {
    if (error_.empty()) {
      error_ = std::move(message);
      std::fprintf(stderr, "[gemv_perf_samples] %s -- writer is fail-closed\n", error_.c_str());
    }
    return false;
  }

  std::FILE* file_ = nullptr;
  bool requested_ = false;
  bool have_run_ = false;
  std::string run_id_;
  std::string error_;
  std::unordered_map<std::string, State> states_;
};

}  // namespace gemv_perf_samples
