#pragma once

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace testfw
{
    class TestFailure : public std::exception
    {
    public:
        explicit TestFailure(std::string message) : message_(std::move(message)) {}
        const char *what() const noexcept override { return message_.c_str(); }

    private:
        std::string message_;
    };

    struct TestCase
    {
        std::string suite;
        std::string name;
        std::function<void()> fn;
    };

    inline std::vector<TestCase> &registry()
    {
        static std::vector<TestCase> tests;
        return tests;
    }

    inline bool registerTest(const char *suite, const char *name, std::function<void()> fn)
    {
        registry().push_back(TestCase{suite, name, std::move(fn)});
        return true;
    }

    inline std::string composeFailureMessage(const char *file,
                                             int line,
                                             const std::string &message)
    {
        std::ostringstream oss;
        oss << file << ':' << line << ": " << message;
        return oss.str();
    }

    inline bool containsFilter(const std::string &candidate, const std::string &filter)
    {
        return filter.empty() || candidate.find(filter) != std::string::npos;
    }

    inline int runAll(const std::string &filter)
    {
        int total = 0;
        int failed = 0;

        for (const TestCase &tc : registry())
        {
            const std::string fullName = tc.suite + "/" + tc.name;
            if (!containsFilter(fullName, filter))
                continue;

            ++total;
            try
            {
                tc.fn();
                std::cout << "[PASS] " << fullName << '\n';
            }
            catch (const std::exception &e)
            {
                ++failed;
                std::cout << "[FAIL] " << fullName << " -> " << e.what() << '\n';
            }
            catch (...)
            {
                ++failed;
                std::cout << "[FAIL] " << fullName << " -> unknown exception\n";
            }
        }

        if (total == 0)
        {
            std::cout << "No tests were selected."
                      << (filter.empty() ? "" : " Filter: " + filter)
                      << '\n';
            return 1;
        }

        std::cout << "\nExecuted " << total << " tests; failures: " << failed << '\n';
        return (failed == 0) ? 0 : 1;
    }
}

#define TEST_CASE(SUITE, NAME)                                                                               \
    static void test_##SUITE##_##NAME();                                                                     \
    static const bool reg_##SUITE##_##NAME = testfw::registerTest(#SUITE, #NAME, test_##SUITE##_##NAME);   \
    static void test_##SUITE##_##NAME()

#define TEST_FAIL(MSG) throw testfw::TestFailure(testfw::composeFailureMessage(__FILE__, __LINE__, (MSG)))

#define CHECK(EXPR)                                                                                          \
    do                                                                                                       \
    {                                                                                                        \
        if (!(EXPR))                                                                                         \
        {                                                                                                    \
            TEST_FAIL(std::string("Check failed: ") + #EXPR);                                              \
        }                                                                                                    \
    } while (false)

#define CHECK_EQ(A, B)                                                                                       \
    do                                                                                                       \
    {                                                                                                        \
        const auto _a = (A);                                                                                 \
        const auto _b = (B);                                                                                 \
        if (!(_a == _b))                                                                                     \
        {                                                                                                    \
            std::ostringstream _oss;                                                                         \
            _oss << "Expected equality: " << #A << " == " << #B << ", got " << _a << " vs " << _b;      \
            TEST_FAIL(_oss.str());                                                                           \
        }                                                                                                    \
    } while (false)

#define CHECK_NEAR(A, B, EPS)                                                                                \
    do                                                                                                       \
    {                                                                                                        \
        const double _a = static_cast<double>(A);                                                            \
        const double _b = static_cast<double>(B);                                                            \
        const double _eps = static_cast<double>(EPS);                                                        \
        if (std::fabs(_a - _b) > _eps)                                                                       \
        {                                                                                                    \
            std::ostringstream _oss;                                                                         \
            _oss << "Expected near: " << #A << " ~= " << #B << " (eps=" << _eps                          \
                 << "), got " << _a << " vs " << _b;                                                       \
            TEST_FAIL(_oss.str());                                                                           \
        }                                                                                                    \
    } while (false)

#define CHECK_THROWS(EXPR)                                                                                   \
    do                                                                                                       \
    {                                                                                                        \
        bool _didThrow = false;                                                                              \
        try                                                                                                  \
        {                                                                                                    \
            (void)(EXPR);                                                                                    \
        }                                                                                                    \
        catch (...)                                                                                          \
        {                                                                                                    \
            _didThrow = true;                                                                                \
        }                                                                                                    \
        if (!_didThrow)                                                                                      \
        {                                                                                                    \
            TEST_FAIL(std::string("Expected exception from: ") + #EXPR);                                   \
        }                                                                                                    \
    } while (false)
