package com.example;

import soot.Body;
import soot.PackManager;
import soot.Transform;
import soot.Unit;
import soot.Scene;
import soot.SceneTransformer;
import soot.SootMethod;
import soot.jimple.InvokeExpr;
import soot.jimple.Jimple;
import soot.jimple.Stmt;
import soot.options.Options;
import soot.util.Chain;
import soot.*;
import soot.jimple.*;
import soot.jimple.internal.*;
import soot.util.*;
import soot.BodyTransformer;
import soot.ValueBox;
import soot.Local;
import soot.jimple.AssignStmt;
import soot.jimple.IntConstant; 
import soot.jimple.StringConstant; 
import soot.tagkit.Tag;    
import soot.SootMethodRef;       
import soot.tagkit.LineNumberTag; // 如果版本支持

import java.util.Arrays;
import java.util.List; 
import java.util.ArrayList; 
import java.util.Iterator;
import java.util.Map;
import java.util.*;


public class VariableInstrumenter extends BodyTransformer {

    private static class VarDef {
        final String varId;
        final String methodSig;
        final int sourceType;
        final int defLine;

        VarDef(String varId, String methodSig, int sourceType, int defLine) {
            this.varId = varId;
            this.methodSig = methodSig;
            this.sourceType = sourceType;
            this.defLine = defLine;
        }
    }

    @Override
    protected void internalTransform(Body body, String phase, Map<String, String> options) {
        SootMethod method = body.getMethod();
        String methodSig = method.getSignature();
        PatchingChain<Unit> units = body.getUnits();
        
        // 每个方法处理使用独立的映射表
        Map<Local, VarDef> varDefMap = new HashMap<>();

        processParameters(method, units, varDefMap);
        processLocalVariables(method, units, varDefMap);
        processVariableUse(methodSig, units, varDefMap);
    }

    private void processParameters(SootMethod method, PatchingChain<Unit> units, 
                                  Map<Local, VarDef> varDefMap) {
        List<Unit> paramUnits = new ArrayList<>();
        Iterator<Unit> iter = units.snapshotIterator();

        // 获取方法签名对应的行号
        int methodLine = method.getJavaSourceStartLineNumber();
        methodLine = methodLine > 0 ? methodLine : 0;  // 处理无行号情况
        // 参数索引计数器
        int paramIndex = 0;

        while (iter.hasNext()) {
            Unit unit = iter.next();
            if (isParameterDefinition(unit)) {
                IdentityStmt stmt = (IdentityStmt) unit;
                Local param = (Local) stmt.getLeftOp();
                
                // 生成带参数位置的sourceType
                VarDef def = new VarDef(
                    generateParamId(param),
                    method.getSignature(),
                    ++paramIndex,  // 参数位置从1开始递增
                    methodLine     // 使用函数头行号
                );
                
                varDefMap.put(param, def);
                insertDefLog(units, unit, def);
            }
        }

    }

    private void processLocalVariables(SootMethod method, PatchingChain<Unit> units,
                                    Map<Local, VarDef> varDefMap) {
        Set<Local> processedLocals = new HashSet<>();
        Iterator<Unit> iter = units.snapshotIterator();
        
        while (iter.hasNext()) {
            Unit unit = iter.next();
            // 处理显式赋值
            if (unit instanceof AssignStmt) {
                AssignStmt assign = (AssignStmt) unit;
                Value left = assign.getLeftOp();
                if (left instanceof Local) {
                    Local local = (Local) left;
                    recordLocalDef(local, unit, method, units, varDefMap);
                    processedLocals.add(local);
                }
            }
            // 处理隐式初始化 (如 i2=0)
            else if (unit instanceof IdentityStmt) {
                IdentityStmt stmt = (IdentityStmt) unit;
                if (stmt.getRightOp() instanceof CaughtExceptionRef) {
                    Local local = (Local) stmt.getLeftOp();
                    if (!processedLocals.contains(local)) {
                        recordLocalDef(local, unit, method, units, varDefMap);
                    }
                }
            }
        }
    }

    private void recordLocalDef(Local local, Unit unit, SootMethod method,
                            PatchingChain<Unit> units, Map<Local, VarDef> varDefMap) {
        String varId = generateLocalId(local);
        int defLine = getValidLineNumber(unit, units);
        VarDef def = new VarDef(varId, method.getSignature(), 0, defLine);
        varDefMap.put(local, def);
        insertDefLog(units, unit, def);
    }

    private void processVariableUse(String currentMethodSig, PatchingChain<Unit> units,
                                Map<Local, VarDef> varDefMap) {
        Iterator<Unit> iter = units.snapshotIterator();
        while (iter.hasNext()) {
            Unit unit = iter.next();
            
            // 处理所有UseBox中的变量
            for (ValueBox vBox : unit.getUseBoxes()) {
                Value value = vBox.getValue();
                if (value instanceof Local) {
                    Local local = (Local) value;
                    VarDef def = varDefMap.get(local);
                    if (def != null) {
                        String calleeSig = "";
                        int paramIndex = 0; // 默认非参数使用
                        // 检测方法调用并计算参数位置
                        if (isMethodInvocation(unit)) {
                            InvokeExpr invokeExpr = getInvokeExpr(unit);
                            calleeSig = invokeExpr.getMethodRef().getSignature();
                            
                            // 遍历参数列表查找变量位置
                            List<Value> args = invokeExpr.getArgs();
                            for (int i = 0; i < args.size(); i++) {
                                if (args.get(i).equals(local)) {
                                    paramIndex = i + 1; // 参数索引从1开始
                                    break;
                                }
                            }
                        }
                        
                        // 插入带有参数位置的日志
                        insertUseLog(units, unit, def, currentMethodSig, calleeSig, paramIndex);
                    }
                }
            }
        }
    }
    // 检测单元是否为方法调用
    private boolean isMethodInvocation(Unit unit) {
        return (unit instanceof InvokeStmt) || 
            (unit instanceof AssignStmt && 
                ((AssignStmt) unit).getRightOp() instanceof InvokeExpr);
    }
    // 提取调用表达式
    private InvokeExpr getInvokeExpr(Unit unit) {
        if (unit instanceof InvokeStmt) {
            return ((InvokeStmt) unit).getInvokeExpr();
        } else if (unit instanceof AssignStmt) {
            return (InvokeExpr) ((AssignStmt) unit).getRightOp();
        }
        return null;
    }


    // private void processValue(Value value, Unit contextUnit, String currentMethodSig,
    //                         PatchingChain<Unit> units, Map<Local, VarDef> varDefMap) {
    //     if (value instanceof Local) {
    //         Local local = (Local) value;
    //         VarDef def = varDefMap.get(local);
    //         if (def != null) {
    //             insertUseLog(units, contextUnit, def, currentMethodSig, "");
    //         }
    //     }
    // }

    // private void handleInvocation(InvokeStmt invoke, String callerMethodSig, 
    //                              PatchingChain<Unit> units, Map<Local, VarDef> varDefMap) {
    //     InvokeExpr expr = invoke.getInvokeExpr();
    //     String calleeSig = expr.getMethodRef().getSignature();
    //     // 过滤日志类自身方法
    //     if (expr.getMethod().getDeclaringClass().getName().startsWith("com.example.DynamicLogger")) {
    //         return;
    //     }

    //     for (Value arg : expr.getArgs()) {
    //         if (arg instanceof Local) {
    //             VarDef def = varDefMap.get(arg);
    //             if (def != null) {
    //                 insertUseLog(units, invoke, def, callerMethodSig, calleeSig);
    //             }
    //         }
    //     }
    // }

    // private void handleAssignment(AssignStmt assign, String callerMethodSig,
    //                              PatchingChain<Unit> units, Map<Local, VarDef> varDefMap) {
    //     if (assign.getRightOp() instanceof InvokeExpr) {
    //         InvokeExpr expr = (InvokeExpr) assign.getRightOp();
    //         String calleeSig = expr.getMethodRef().getSignature();
    //         // 过滤日志类自身方法
    //         if (expr.getMethod().getDeclaringClass().getName().startsWith("com.example.DynamicLogger")) {
    //             return;
    //         }

    //         for (Value arg : expr.getArgs()) {
    //             if (arg instanceof Local) {
    //                 VarDef def = varDefMap.get(arg);
    //                 if (def != null) {
    //                     insertUseLog(units, assign, def, callerMethodSig, calleeSig,0);
    //                 }
    //             }
    //         }
    //     }
    // }

    private String generateParamId(Local param) {
        return param.getName() + "@" + System.identityHashCode(param);
    }

    private String generateLocalId(Local local) {
        return local.getName() + "@" + System.identityHashCode(local);
    }


    private boolean isParameterDefinition(Unit unit) {
        return unit instanceof IdentityStmt 
            && ((IdentityStmt)unit).getRightOp() instanceof ParameterRef;
    }

    private void insertDefLog(PatchingChain<Unit> units, Unit position, VarDef def) {
        Unit logStmt = Jimple.v().newInvokeStmt(
            Jimple.v().newStaticInvokeExpr(
                Scene.v().getMethod(
                    "<com.example.DynamicLogger: void logDef(java.lang.String,java.lang.String,int,int)>"
                ).makeRef(),
                Arrays.asList(
                    StringConstant.v(def.varId),
                    StringConstant.v(def.methodSig),
                    IntConstant.v(def.sourceType),
                    IntConstant.v(def.defLine)
                )
            )
        );
        units.insertBefore(logStmt, position);
    }

    private void insertUseLog(PatchingChain<Unit> units, Unit position, 
                            VarDef def, String currentMethodSig, String calleeSig,int paramIndex) {
        // 过滤日志类自身的方法调用
        if (calleeSig.startsWith("<com.example.DynamicLogger:")) {
            return;
        }
        
        // 非参数使用时清空calleeSig
        if (paramIndex == 0) {
            calleeSig = "";
        }
        Unit logStmt = Jimple.v().newInvokeStmt(
            Jimple.v().newStaticInvokeExpr(
                Scene.v().getMethod(
                    "<com.example.DynamicLogger: void logUse(java.lang.String,java.lang.String,java.lang.String,int,int)>"
                ).makeRef(),
                Arrays.asList(
                    StringConstant.v(def.varId),
                    StringConstant.v(currentMethodSig),
                    StringConstant.v(calleeSig),
                    IntConstant.v(paramIndex), 
                    IntConstant.v(getValidLineNumber(position,units))
                )
            )
        );
        units.insertBefore(logStmt, position);
    }

    private int getLineNumber(Unit unit) {
        return unit.getJavaSourceStartLineNumber() > 0 
             ? unit.getJavaSourceStartLineNumber() 
             : 0;
    }

        // 获取有效的行号信息（向前/向后查找）
    private int getValidLineNumber(Unit unit, Chain<Unit> units) {
        Unit current = unit;

        // 向后查找
        while (current != null) {
            for (Tag tag : current.getTags()) {
                if (tag instanceof LineNumberTag) {
                    return ((LineNumberTag) tag).getLineNumber();
                }
            }
            current = units.getSuccOf(current);
        }
        // 向前查找 
        current = unit;
        while (current != null) {
            for (Tag tag : current.getTags()) {
                if (tag instanceof LineNumberTag) {
                    return ((LineNumberTag) tag).getLineNumber();
                }
            }
            current = units.getPredOf(current);
        }
        return 0;
    }


}
