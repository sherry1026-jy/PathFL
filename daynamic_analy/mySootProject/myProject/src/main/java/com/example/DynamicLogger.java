package com.example;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.ArrayDeque;
import java.util.Deque;
import java.io.FileWriter;
import java.util.*;
import java.util.concurrent.*;
import java.io.*;
// import java.time.Instant;                 // <-- 移除（JDK8）
// import java.util.function.Consumer;        // <-- 移除（JDK8）
import java.nio.file.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.io.BufferedWriter;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class DynamicLogger {
    // 记录方法调用
    public static void logMethodCall(String caller, String callee, int line) {
        try {
            // 使用绝对路径，避免权限问题
            //caller+ "|" + callee 表示所在函数|调用函数
            Files.write(Paths.get("/tmp/instrument.log"),
                       ("[CALL] "+line + "|"+caller+ "|" + callee +"\n").getBytes(),
                       StandardOpenOption.CREATE,
                       StandardOpenOption.APPEND);
        } catch (IOException e) {
            e.printStackTrace();
        }
        
    }

    public static void logMethodEntry(String callee, int line) {
        try {
            // 使用绝对路径，避免权限问题 
            Files.write(Paths.get("/tmp/instrument.log"),
                       ("[Entry] "+line + "|"+callee+"\n").getBytes(),
                       StandardOpenOption.CREATE,
                       StandardOpenOption.APPEND);
        } catch (IOException e) {
            e.printStackTrace();
        }
        //System.out.println("[CALL] " + caller + " -> " + callee);
    }

    public static void logMethodExit(String callee, int line) {
        try {
            // 使用绝对路径，避免权限问题
            Files.write(Paths.get("/tmp/instrument.log"),
                       ("[Exit] " + line +" |" + callee +"\n").getBytes(),
                       StandardOpenOption.CREATE,
                       StandardOpenOption.APPEND);
        } catch (IOException e) {
            e.printStackTrace();
        }
        //System.out.println("[CALL] " + caller + " -> " + callee);
    }

    // 记录变量定义（保持原样）
    public static void logDef(String varName, String originMethod, int sourceType, int line) {
        String logMsg = String.format("[DEF] %d | @%s | origin=%s | Type=%s\n",
                line,varName, originMethod, sourceType);
        writeLog(logMsg);
    }

    // 记录变量使用（新增calleeMethod参数）
    public static void logUse(String varName, String currentMethod, String calleeMethod,int sourceType, int line) {
        String logMsg = String.format("[USE] %d | @%s | current=%s | target=%s | Type=%s\n",
                line, varName,  currentMethod, calleeMethod, sourceType);
        writeLog(logMsg);
    }

    // 公共日志写入方法
    private static void writeLog(String message) {
        try {
            Files.write(Paths.get("/tmp/instrument.log"),
                    message.getBytes(),
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void logBranch(int conditionLine, String methodSig, int targetLine) {
        try {
            // 使用绝对路径，避免权限问题
            String logMessage = String.format("[CFG]%d→%d|%s\n",conditionLine,targetLine,methodSig);
            Files.write(Paths.get("/tmp/instrument.log"),
                       logMessage.getBytes(),
                       StandardOpenOption.CREATE,
                       StandardOpenOption.APPEND);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void logCFG(int conditionLine, String methodSig, int targetLine) {
        try {
            // 使用绝对路径，避免权限问题
            String logMessage = String.format("[CFG]%d→%d|%s\n",conditionLine,targetLine,methodSig);
            Files.write(Paths.get("/tmp/instrument.log"),
                       logMessage.getBytes(),
                       StandardOpenOption.CREATE,
                       StandardOpenOption.APPEND);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
