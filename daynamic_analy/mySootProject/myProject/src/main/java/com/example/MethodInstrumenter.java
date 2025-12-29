package com.example;

import soot.Body;
import soot.BodyTransformer;
import soot.PatchingChain;
import soot.Scene;
import soot.SootMethod;
import soot.SootMethodRef;
import soot.Unit;
import soot.jimple.AssignStmt;
import soot.jimple.GotoStmt;
import soot.jimple.IdentityStmt;
import soot.jimple.IfStmt;
import soot.jimple.IntConstant;
import soot.jimple.InvokeExpr;
import soot.jimple.InvokeStmt;
import soot.jimple.Jimple;
import soot.jimple.NopStmt;
import soot.jimple.ReturnStmt;
import soot.jimple.ReturnVoidStmt;
import soot.jimple.Stmt;
import soot.jimple.StringConstant;
import soot.jimple.ThrowStmt;
import soot.tagkit.LineNumberTag;
import soot.tagkit.Tag;
import soot.util.Chain;

import java.util.*;

public class MethodInstrumenter extends BodyTransformer {

    @Override
    protected void internalTransform(Body body, String phase, Map<String, String> options) {
        SootMethod method = body.getMethod();
        String methodSig = method.getSignature();
        PatchingChain<Unit> units = body.getUnits();

        int lastExecutedLine = -1;

        // —— 方法入口插桩（修复“方法头行号”）——
        int methodEntryLine = insertMethodEntryLog(body, units, methodSig, method);

        Iterator<Unit> unitIterator = units.snapshotIterator();
        while (unitIterator.hasNext()) {
            Unit unit = unitIterator.next();

            // 顺序流
            lastExecutedLine = processSequenceFlow(units, unit, methodSig, lastExecutedLine, methodEntryLine);

            // 方法调用
            if (unit instanceof InvokeStmt) {
                processInvokeStmt(units, (InvokeStmt) unit, methodSig);
            } else if (unit instanceof AssignStmt) {
                processAssignStmt(units, (AssignStmt) unit, methodSig);
            }

            // 方法退出
            if (unit instanceof ReturnStmt || unit instanceof ReturnVoidStmt) {
                insertExitLog(units, unit, methodSig);
            }

            // 条件分支
            if (unit instanceof IfStmt) {
                processConditionalBranch(body, units, (IfStmt) unit, methodSig);
            }
        }
    }

    /** 计算“方法头行号”：优先 Java 源起始行；否则用方法内最小行号；最后退化为 0 */
    private int computeMethodHeaderLine(SootMethod method, Chain<Unit> units) {
        int h = method.getJavaSourceStartLineNumber();
        if (h > 0) return h;

        int min = Integer.MAX_VALUE;
        for (Unit u : units) {
            for (Tag t : u.getTags()) {
                if (t instanceof LineNumberTag) {
                    min = Math.min(min, ((LineNumberTag) t).getLineNumber());
                }
            }
        }
        return (min == Integer.MAX_VALUE) ? 0 : min;
    }

    /** 方法入口日志：用“方法头行号”且补一条 header -> 首条可执行行 的 CFG 边 */
    private int insertMethodEntryLog(Body body, Chain<Unit> units, String methodSig, SootMethod method) {
        Unit firstUnit = findSafeInsertPoint(units.getFirst(), units);
        int headerLine = computeMethodHeaderLine(method, units);

        Stmt entryLog = Jimple.v().newInvokeStmt(
                Jimple.v().newStaticInvokeExpr(
                        Scene.v().getMethod("<com.example.DynamicLogger: void logMethodEntry(java.lang.String,int)>").makeRef(),
                        Arrays.asList(StringConstant.v(methodSig), IntConstant.v(headerLine))
                )
        );
        units.insertBefore(entryLog, firstUnit);

        // 入口 CFG：方法头 -> 首条有效行（若二者不同且均有效）
        int firstLine = getValidLineNumber(firstUnit, units);
        if (headerLine > 0 && firstLine > 0 && headerLine != firstLine) {
            insertBranchLog(units, firstUnit, headerLine, firstLine, methodSig);
        }
        return headerLine;
    }

    private void processInvokeStmt(Chain<Unit> units, InvokeStmt invoke, String methodSig) {
        SootMethod callee = invoke.getInvokeExpr().getMethod();
        if (shouldSkipMethod(callee)) return;

        int line = getValidLineNumber(invoke, units);
        Stmt callLog = Jimple.v().newInvokeStmt(
                Jimple.v().newStaticInvokeExpr(
                        Scene.v().getMethod("<com.example.DynamicLogger: void logMethodCall(java.lang.String,java.lang.String,int)>").makeRef(),
                        Arrays.asList(
                                StringConstant.v(methodSig),
                                StringConstant.v(callee.getSignature()),
                                IntConstant.v(line)
                        )
                )
        );
        units.insertBefore(callLog, invoke);
    }

    private void processAssignStmt(Chain<Unit> units, AssignStmt assign, String methodSig) {
        if (!(assign.getRightOp() instanceof InvokeExpr)) return;

        InvokeExpr invokeExpr = (InvokeExpr) assign.getRightOp();
        SootMethod callee = invokeExpr.getMethod();
        if (shouldSkipMethod(callee)) return;

        int line = getValidLineNumber(assign, units);
        Stmt callLog = Jimple.v().newInvokeStmt(
                Jimple.v().newStaticInvokeExpr(
                        Scene.v().getMethod("<com.example.DynamicLogger: void logMethodCall(java.lang.String,java.lang.String,int)>").makeRef(),
                        Arrays.asList(
                                StringConstant.v(methodSig),
                                StringConstant.v(callee.getSignature()),
                                IntConstant.v(line)
                        )
                )
        );
        units.insertBefore(callLog, assign);
    }

    private void insertExitLog(Chain<Unit> units, Unit unit, String methodSig) {
        int line = getValidLineNumber(unit, units);
        Stmt exitLog = Jimple.v().newInvokeStmt(
                Jimple.v().newStaticInvokeExpr(
                        Scene.v().getMethod("<com.example.DynamicLogger: void logMethodExit(java.lang.String,int)>").makeRef(),
                        Arrays.asList(StringConstant.v(methodSig), IntConstant.v(line))
                )
        );
        units.insertBefore(exitLog, unit);
    }

    private void processConditionalBranch(Body body, Chain<Unit> units, IfStmt ifStmt, String methodSig) {
        int condLine = getValidLineNumber(ifStmt, units);
        if (condLine == -1) return;

        // 真分支
        Unit trueTarget = findRealTarget(ifStmt.getTarget(), units);
        int trueLine = getValidLineNumber(trueTarget, units);

        // 假分支（fall-through）
        Unit falseTarget = findRealTarget(units.getSuccOf(ifStmt), units);
        int falseLine = getValidLineNumber(falseTarget, units);

        if (trueLine != -1 && trueLine != condLine) {
            insertBranchLog(units, ifStmt.getTarget(), condLine, trueLine, methodSig);
        }
        if (falseLine != -1 && falseLine != condLine) {
            insertBranchLog(units, units.getSuccOf(ifStmt), condLine, falseLine, methodSig);
        }
    }

    private Unit findRealTarget(Unit start, Chain<Unit> units) {
        Unit current = start;
        while (current != null) {
            if (!(current instanceof NopStmt) && !isLogStatement(current)) {
                return current;
            }
            current = units.getSuccOf(current);
        }
        return start;
    }

    private boolean isLogStatement(Unit unit) {
        if (unit instanceof InvokeStmt) {
            SootMethodRef ref = ((InvokeStmt) unit).getInvokeExpr().getMethodRef();
            return ref.getDeclaringClass().getName().equals("com.example.DynamicLogger");
        }
        return false;
    }

    private void insertBranchLog(Chain<Unit> units, Unit target, int condLine, int targetLine, String methodSig) {
        if (condLine == targetLine) return;
        Stmt logStmt = Jimple.v().newInvokeStmt(
                Jimple.v().newStaticInvokeExpr(
                        Scene.v().getMethod("<com.example.DynamicLogger: void logBranch(int,java.lang.String,int)>").makeRef(),
                        Arrays.asList(
                                IntConstant.v(condLine),
                                StringConstant.v(methodSig),
                                IntConstant.v(targetLine)
                        )
                )
        );
        units.insertBefore(logStmt, target);
    }

    /** 向后→向前各扫一遍，拿到最近可用的行号；失败返回 -1（保持你的原语义） */
    private int getValidLineNumber(Unit unit, Chain<Unit> units) {
        Unit current = unit;
        while (current != null) {
            for (Tag tag : current.getTags()) {
                if (tag instanceof LineNumberTag) {
                    return ((LineNumberTag) tag).getLineNumber();
                }
            }
            current = units.getSuccOf(current);
        }
        current = unit;
        while (current != null) {
            for (Tag tag : current.getTags()) {
                if (tag instanceof LineNumberTag) {
                    return ((LineNumberTag) tag).getLineNumber();
                }
            }
            current = units.getPredOf(current);
        }
        return -1;
    }

    private int processSequenceFlow(Chain<Unit> units, Unit currentUnit, String methodSig,
                                    int lastExecutedLine, int methodEntryLine) {
        if (shouldSkipForSequence(currentUnit)) return lastExecutedLine;
        int currentLine = getLineNumberDirect(currentUnit);
        if (currentLine == -1) return lastExecutedLine;

        Unit nextUnit = findSequentialSuccessor(currentUnit, units);
        if (nextUnit == null) return lastExecutedLine;

        int nextLine = getLineNumberDirect(nextUnit);
        if (nextLine == -1 || nextLine == currentLine) return lastExecutedLine;

        insertSequenceLogAfter(units, currentUnit, currentLine, nextLine, methodSig, methodEntryLine);
        return nextLine;
    }

    private Unit findSequentialSuccessor(Unit unit, Chain<Unit> units) {
        Unit next = units.getSuccOf(unit);
        while (next != null) {
            if (isControlFlowStatement(unit) || isLogStatement(unit)) return null;
            if (getLineNumberDirect(next) != -1) return next;
            next = units.getSuccOf(next);
        }
        return null;
    }

    private boolean isControlFlowStatement(Unit unit) {
        return unit instanceof GotoStmt ||
               unit instanceof IfStmt ||
               unit instanceof ThrowStmt ||
               unit instanceof ReturnStmt ||
               unit instanceof ReturnVoidStmt;
    }

    private int getEnhancedLineNumber(Unit unit, Chain<Unit> units) {
        int line = getLineNumberDirect(unit);
        if (line != -1) return line;
        Unit successor = units.getSuccOf(unit);
        if (successor != null) {
            line = getLineNumberDirect(successor);
            if (line != -1) return line;
        }
        Unit predecessor = units.getPredOf(unit);
        if (predecessor != null) {
            line = getLineNumberDirect(predecessor);
            if (line != -1) return line;
        }
        return -1;
    }

    private int getLineNumberDirect(Unit unit) {
        for (Tag tag : unit.getTags()) {
            if (tag instanceof LineNumberTag) {
                return ((LineNumberTag) tag).getLineNumber();
            }
        }
        return -1;
    }

    private boolean shouldSkipForSequence(Unit unit) {
        return unit instanceof IdentityStmt ||
               unit instanceof NopStmt ||
               isLogStatement(unit) ||
               unit instanceof GotoStmt ||
               unit instanceof ThrowStmt ||
               unit instanceof ReturnStmt ||
               unit instanceof ReturnVoidStmt;
    }

    private void insertSequenceLog(Chain<Unit> units, Unit targetUnit,
                                   int fromLine, int toLine, String methodSig, int methodEntryLine) {
        if (fromLine == -1) fromLine = methodEntryLine;
        if (fromLine == toLine) return;

        Stmt logStmt = Jimple.v().newInvokeStmt(
                Jimple.v().newStaticInvokeExpr(
                        Scene.v().getMethod("<com.example.DynamicLogger: void logBranch(int,java.lang.String,int)>").makeRef(),
                        Arrays.asList(
                                IntConstant.v(fromLine),
                                StringConstant.v(methodSig),
                                IntConstant.v(toLine)
                        )
                )
        );
        Unit insertPoint = findSafeInsertPoint(targetUnit, units);
        units.insertBefore(logStmt, insertPoint);
    }

    private void insertSequenceLogAfter(Chain<Unit> units, Unit currentUnit,
                                        int fromLine, int toLine, String methodSig, int methodEntryLine) {
        if (fromLine == -1) fromLine = methodEntryLine;
        if (fromLine == toLine) return;

        Stmt logStmt = Jimple.v().newInvokeStmt(
                Jimple.v().newStaticInvokeExpr(
                        Scene.v().getMethod("<com.example.DynamicLogger: void logBranch(int,java.lang.String,int)>").makeRef(),
                        Arrays.asList(
                                IntConstant.v(fromLine),
                                StringConstant.v(methodSig),
                                IntConstant.v(toLine)
                        )
                )
        );
        units.insertAfter(logStmt, currentUnit);
    }

    private Unit findSafeInsertPoint(Unit unit, Chain<Unit> units) {
        Unit current = unit;
        while (current instanceof IdentityStmt || current instanceof NopStmt) {
            current = units.getSuccOf(current);
            if (current == null) break;
        }
        return current != null ? current : unit;
    }

    private boolean shouldSkipMethod(SootMethod method) {
        String className = method.getDeclaringClass().getName();
        return className.startsWith("java.") ||
               className.startsWith("javax.") ||
               className.startsWith("sun.") ||
               className.startsWith("com.example.DynamicLogger");
    }
}
