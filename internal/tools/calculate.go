package tools

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"slices"
	"strconv"
	"strings"
)

// mathFuncs maps lower-case function names to their implementations.
// Each function receives evaluated arguments and returns (result, error).
var mathFuncs = map[string]func([]float64) (float64, error){
	"sqrt":  oneArg("sqrt", math.Sqrt),
	"cbrt":  oneArg("cbrt", math.Cbrt),
	"abs":   oneArg("abs", math.Abs),
	"floor": oneArg("floor", math.Floor),
	"ceil":  oneArg("ceil", math.Ceil),
	"round": oneArg("round", math.Round),
	"trunc": oneArg("trunc", math.Trunc),
	"exp":   oneArg("exp", math.Exp),
	"exp2":  oneArg("exp2", math.Exp2),
	"log":   oneArg("log", math.Log),
	"log2":  oneArg("log2", math.Log2),
	"log10": oneArg("log10", math.Log10),
	"sin":   oneArg("sin", math.Sin),
	"cos":   oneArg("cos", math.Cos),
	"tan":   oneArg("tan", math.Tan),
	"asin":  oneArg("asin", math.Asin),
	"acos":  oneArg("acos", math.Acos),
	"atan":  oneArg("atan", math.Atan),
	"sinh":  oneArg("sinh", math.Sinh),
	"cosh":  oneArg("cosh", math.Cosh),
	"tanh":  oneArg("tanh", math.Tanh),
	"pow": func(args []float64) (float64, error) {
		if len(args) != 2 {
			return 0, fmt.Errorf("pow takes 2 arguments, got %d", len(args))
		}
		return math.Pow(args[0], args[1]), nil
	},
	"atan2": func(args []float64) (float64, error) {
		if len(args) != 2 {
			return 0, fmt.Errorf("atan2 takes 2 arguments, got %d", len(args))
		}
		return math.Atan2(args[0], args[1]), nil
	},
	"hypot": func(args []float64) (float64, error) {
		if len(args) != 2 {
			return 0, fmt.Errorf("hypot takes 2 arguments, got %d", len(args))
		}
		return math.Hypot(args[0], args[1]), nil
	},
	"mod": func(args []float64) (float64, error) {
		if len(args) != 2 {
			return 0, fmt.Errorf("mod takes 2 arguments, got %d", len(args))
		}
		if args[1] == 0 {
			return 0, fmt.Errorf("mod: divisor is zero")
		}
		return math.Mod(args[0], args[1]), nil
	},
	"min": func(args []float64) (float64, error) {
		if len(args) != 2 {
			return 0, fmt.Errorf("min takes 2 arguments, got %d", len(args))
		}
		return math.Min(args[0], args[1]), nil
	},
	"max": func(args []float64) (float64, error) {
		if len(args) != 2 {
			return 0, fmt.Errorf("max takes 2 arguments, got %d", len(args))
		}
		return math.Max(args[0], args[1]), nil
	},
}

// mathConstants maps lower-case identifier names to their values.
var mathConstants = map[string]float64{
	"pi":    math.Pi,
	"e":     math.E,
	"phi":   math.Phi,
	"sqrt2": math.Sqrt2,
	"ln2":   math.Ln2,
	"log2e": math.Log2E,
}

func oneArg(name string, fn func(float64) float64) func([]float64) (float64, error) {
	return func(args []float64) (float64, error) {
		if len(args) != 1 {
			return 0, fmt.Errorf("%s takes 1 argument, got %d", name, len(args))
		}
		return fn(args[0]), nil
	}
}

// evalExpression parses and evaluates a mathematical expression string.
// It uses Go's parser so the syntax is Go-like:
//   - Integers: 42, 0xff, 0b1010, 0o17
//   - Floats:   3.14, 1.5e10
//   - Operators: + - * / % (^ is bitwise XOR; use pow(x,y) for exponentiation)
//   - Functions: sqrt, abs, floor, ceil, round, pow, log, sin, cos, tan, …
//   - Constants: pi, e, phi, sqrt2, ln2
func evalExpression(expr string) (float64, error) {
	node, err := parser.ParseExpr(expr)
	if err != nil {
		return 0, fmt.Errorf("parse error: %s", err)
	}
	return evalNode(node)
}

func evalNode(node ast.Expr) (float64, error) {
	switch n := node.(type) {
	case *ast.BasicLit:
		switch n.Kind {
		case token.INT:
			v, err := strconv.ParseInt(n.Value, 0, 64)
			if err != nil {
				// Try uint for large hex values like 0xffffffffffffffff.
				u, err2 := strconv.ParseUint(n.Value, 0, 64)
				if err2 != nil {
					return 0, fmt.Errorf("invalid integer %q: %w", n.Value, err)
				}
				return float64(u), nil
			}
			return float64(v), nil
		case token.FLOAT:
			v, err := strconv.ParseFloat(n.Value, 64)
			if err != nil {
				return 0, fmt.Errorf("invalid float %q: %w", n.Value, err)
			}
			return v, nil
		}
		return 0, fmt.Errorf("unsupported literal kind: %v", n.Kind)

	case *ast.Ident:
		if v, ok := mathConstants[strings.ToLower(n.Name)]; ok {
			return v, nil
		}
		return 0, fmt.Errorf("unknown identifier %q (known constants: pi, e, phi, sqrt2, ln2)", n.Name)

	case *ast.ParenExpr:
		return evalNode(n.X)

	case *ast.UnaryExpr:
		v, err := evalNode(n.X)
		if err != nil {
			return 0, err
		}
		switch n.Op {
		case token.SUB:
			return -v, nil
		case token.ADD:
			return v, nil
		case token.XOR: // ^x is bitwise complement in Go
			return float64(^int64(v)), nil
		}
		return 0, fmt.Errorf("unsupported unary operator %v", n.Op)

	case *ast.BinaryExpr:
		l, err := evalNode(n.X)
		if err != nil {
			return 0, err
		}
		r, err := evalNode(n.Y)
		if err != nil {
			return 0, err
		}
		switch n.Op {
		case token.ADD:
			return l + r, nil
		case token.SUB:
			return l - r, nil
		case token.MUL:
			return l * r, nil
		case token.QUO:
			if r == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			return l / r, nil
		case token.REM:
			if r == 0 {
				return 0, fmt.Errorf("modulo by zero")
			}
			return math.Mod(l, r), nil
		case token.AND:
			return float64(int64(l) & int64(r)), nil
		case token.OR:
			return float64(int64(l) | int64(r)), nil
		case token.XOR:
			return float64(int64(l) ^ int64(r)), nil
		case token.SHL:
			return float64(int64(l) << uint(int64(r))), nil
		case token.SHR:
			return float64(int64(l) >> uint(int64(r))), nil
		case token.AND_NOT:
			return float64(int64(l) &^ int64(r)), nil
		}
		return 0, fmt.Errorf("unsupported operator %v — use pow(x,y) for exponentiation", n.Op)

	case *ast.CallExpr:
		fn, ok := n.Fun.(*ast.Ident)
		if !ok {
			return 0, fmt.Errorf("only simple function calls are supported (e.g. sqrt(x))")
		}
		impl, ok := mathFuncs[strings.ToLower(fn.Name)]
		if !ok {
			names := make([]string, 0, len(mathFuncs))
			for k := range mathFuncs {
				names = append(names, k)
			}
			slices.Sort(names)
			return 0, fmt.Errorf("unknown function %q (available: %s)", fn.Name, strings.Join(names, ", "))
		}
		args := make([]float64, len(n.Args))
		for i, arg := range n.Args {
			v, err := evalNode(arg)
			if err != nil {
				return 0, fmt.Errorf("argument %d of %s: %w", i+1, fn.Name, err)
			}
			args[i] = v
		}
		return impl(args)
	}

	return 0, fmt.Errorf("unsupported expression type %T", node)
}

// formatResult formats a float64 result for display.
// Whole numbers are shown without a decimal point when they fit in an int64;
// otherwise strconv chooses the shortest faithful representation.
func formatResult(v float64) string {
	if math.IsInf(v, 1) {
		return "+Inf"
	}
	if math.IsInf(v, -1) {
		return "-Inf"
	}
	if math.IsNaN(v) {
		return "NaN"
	}
	if v == math.Trunc(v) && math.Abs(v) < 1e15 {
		return strconv.FormatInt(int64(v), 10)
	}
	return strconv.FormatFloat(v, 'g', -1, 64)
}
