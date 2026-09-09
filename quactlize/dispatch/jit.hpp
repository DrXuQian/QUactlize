#pragma once
#include "policy.hpp"
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <spawn.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char** environ;

namespace quactlize::dispatch {
struct Jit {
    std::string python, helper, sdk, cache, source;
    bool enabled() const { return !helper.empty(); }
};

// posix_spawn, not fork or a shell: query may run in a multi-threaded model
// loader. The child does CPU compilation only. Diagnostics go to stderr;
// stdout carries a bounded receipt, never source code or a library path.
inline std::string compile_parent(Jit const& jit, Config const& c) {
    std::vector<std::string> args{
        jit.python, jit.helper, "resolve", "--sdk", jit.sdk, "--cache", jit.cache,
        "--source-contract", jit.source,
        "--parent", c.symbol, "--tuple", std::to_string(c.qtype), std::to_string(c.route),
        std::to_string(c.tm), std::to_string(c.tn), std::to_string(c.tk),
        std::to_string(c.wm), std::to_string(c.wn), std::to_string(c.stages),
        std::to_string(c.ap), std::to_string(c.dn), std::to_string(c.parent_persistent)};
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    int fd[2];
    if (pipe2(fd, O_CLOEXEC)) throw std::runtime_error("JIT receipt pipe failed");
    posix_spawn_file_actions_t actions;
    int rc=posix_spawn_file_actions_init(&actions);
    if (rc) { close(fd[0]); close(fd[1]); throw std::runtime_error("JIT spawn setup failed"); }
    rc=posix_spawn_file_actions_adddup2(&actions,fd[1],STDOUT_FILENO);
    if (!rc) rc=posix_spawn_file_actions_addclose(&actions,fd[0]);
    if (!rc) rc=posix_spawn_file_actions_addclose(&actions,fd[1]);
    pid_t pid=-1;
    if (!rc) rc=posix_spawn(&pid,jit.python.c_str(),&actions,nullptr,argv.data(),environ);
    posix_spawn_file_actions_destroy(&actions);
    close(fd[1]);
    if (rc) { close(fd[0]); throw std::runtime_error(std::string("JIT spawn: ")+std::strerror(rc)); }
    std::string result;
    char buffer[512];
    bool overflow=false, read_error=false;
    for (;;) {
        ssize_t count=read(fd[0],buffer,sizeof(buffer));
        if (count<0 && errno==EINTR) continue;
        if (count<0) read_error=true;
        if (count<=0) break;
        if (result.size()+size_t(count)>4096) overflow=true;
        else if (!overflow) result.append(buffer,size_t(count));
    }
    close(fd[0]);
    int status=0;
    while (waitpid(pid,&status,0)<0) {
        if (errno==EINTR) continue;
        throw std::runtime_error("JIT wait failed");
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status))
        throw std::runtime_error("selected parent JIT failed; see compiler diagnostic/build.log");
    if (overflow || read_error) throw std::runtime_error("JIT receipt unreadable/oversized");
    return result;
}
inline bool hex_digest(std::string const& value) {
    return value.size()==64 && value.find_first_not_of("0123456789abcdef")==std::string::npos;
}
} // namespace quactlize::dispatch
