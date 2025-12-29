package com.example;

import soot.*;
import soot.options.Options;
import soot.PackManager;
import soot.Transform;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.*;

public class App {

    public static void main(String[] args) throws IOException {
        final String defects4jBase = "/home/vipuser/test/defect_data/data120/Closure/";
        final String myLoggerClass  = "/home/vipuser/test/mySootProject/myProject/build/classes/com/example/DynamicLogger.class";
        final String myLoggerSource = "/home/vipuser/test/mySootProject/myProject/src/main/java/com/example/DynamicLogger.java";

        for (int n = 44; n <= 44; n++) {
            String projectPath = defects4jBase + n + "/Closure_" + n + "_buggy";
            String srcListFile = defects4jBase + n + "/Closure_" + n + "_src"; // 每行一个FQN（外部类）
            String classesDir  = projectPath + "/build/classes";
            // String classesDir  = projectPath + "/target/classes";
            String overlayDir  = defects4jBase + n + "/overlay_Closure_" + n;
            String testpath ="/home/vipuser/test/mySootProject/myProject/sootOutput";

            if (!Files.exists(Paths.get(srcListFile))) continue;
            copyLoggerFiles(projectPath, myLoggerClass, myLoggerSource);

            // —— JDK7：不用 stream，手动读 & 过滤
            List<String> raw = Files.readAllLines(Paths.get(srcListFile), StandardCharsets.UTF_8);
            List<String> outerFqns = new ArrayList<String>();
            for (String s : raw) {
                if (s != null) {
                    s = s.trim();
                    if (!s.isEmpty()) outerFqns.add(s);
                }
            }

            // 扩展内部类（Outer$*）
            Set<String> whiteList = expandWithInnerClasses(classesDir, outerFqns);

            // 仅用于缩小解析范围的包白名单
            Set<String> includePkgs = new LinkedHashSet<String>();
            for (String fqn : whiteList) {
                String pkg = pkgPrefixOf(fqn);
                if (!pkg.isEmpty()) includePkgs.add(pkg);
            }

            // ==== 构建 Soot classpath：项目 classes + 所有 lib/*.jar（递归） ====
            String sootCp = buildSootClasspath(classesDir, projectPath);

            // ========= Soot 配置 =========
            G.reset();
            Options.v().set_prepend_classpath(true);
            Options.v().set_allow_phantom_refs(true);
            Options.v().set_keep_line_number(true);
            Options.v().set_derive_java_version(true);     // （可选）若你想 49，用这一行替代上行
            Options.v().set_jasmin_backend(true);
            Options.v().set_src_prec(Options.src_prec_only_class);
            Options.v().set_output_format(Options.output_format_class);

            Options.v().set_output_dir(classesDir);                        // 原地覆盖
            Options.v().set_process_dir(Collections.singletonList(classesDir)); // 只处理工程产出的 class
            Options.v().set_soot_classpath(sootCp);                        // ⭐ 关键：补全依赖
            Options.v().set_no_bodies_for_excluded(true);
            Options.v().set_exclude(Arrays.asList("java.", "javax.", "sun.", "com.sun.", "jdk."));
            Options.v().setPhaseOption("jb", "use-original-names:true");

            if (!includePkgs.isEmpty()) {
                List<String> includeList = new ArrayList<String>();
                for (String p : includePkgs) {
                    String q = p.endsWith(".") ? p : (p + ".");
                    if (!includeList.contains(q)) includeList.add(q);
                }
                Options.v().set_include(includeList);
                System.out.println("[Soot] include packages = " + includeList);
            } else {
                System.out.println("[Soot] no include packages set (relying on process_dir + application marking)");
            }

            // 注册变换器（只作用于 application 类）
            PackManager.v().getPack("jtp").add(new Transform("jtp.varTransform",    new VariableInstrumenter()));
            PackManager.v().getPack("jtp").add(new Transform("jtp.methodTransform", new MethodInstrumenter()));

            // 避免对 Logger 插桩
            SootClass loggerClass = Scene.v().loadClassAndSupport("com.example.DynamicLogger");
            loggerClass.setLibraryClass();

            // 触发加载
            Scene.v().loadNecessaryClasses();

            // 全部先标为 library；白名单改为 application
            List<SootClass> all = new ArrayList<SootClass>(Scene.v().getClasses());
            for (SootClass c : all) c.setLibraryClass();
            for (String fqn : whiteList) {
                SootClass c = Scene.v().containsClass(fqn)
                        ? Scene.v().getSootClass(fqn)
                        : Scene.v().loadClassAndSupport(fqn);
                c.setApplicationClass();
            }

            PackManager.v().runPacks();
            PackManager.v().writeOutput();

            System.out.println("[OK] instrumented " + whiteList.size() + " classes (outer + inners)");
            System.out.println("     classesDir: " + classesDir);
            System.out.println("     soot-classpath jars: " + countJarsInClasspath(sootCp));
        }
    }

    /** 构建 soot-classpath：classesDir + {projectPath/lib/**, projectPath/build/lib/**, third_party/**} 下的所有 .jar */
    private static String buildSootClasspath(String classesDir, String projectPath) throws IOException {
        List<String> cp = new ArrayList<String>();
        cp.add(classesDir);

        List<Path> libRoots = Arrays.asList(
                Paths.get(projectPath, "lib"),
                Paths.get(projectPath, "build", "lib"),
                Paths.get(projectPath, "third_party"),
                Paths.get(projectPath, "build", "third_party")
        );

        for (Path root : libRoots) {
            cp.addAll(collectJars(root)); // JDK7 递归遍历
        }

        // 加上当前进程 classpath（含 JRE/插件等）
        cp.add(System.getProperty("java.class.path", ""));

        // 去重并拼接（JDK7 没有 String.join）
        LinkedHashSet<String> dedup = new LinkedHashSet<String>();
        for (String e : cp) if (e != null && !e.isEmpty()) dedup.add(e);
        StringBuilder sb = new StringBuilder();
        for (String e : dedup) {
            if (sb.length() > 0) sb.append(File.pathSeparatorChar);
            sb.append(e);
        }
        return sb.toString();
    }

    /** JDK7：用 walkFileTree 递归收集 .jar */
    private static List<String> collectJars(Path root) throws IOException {
        final List<String> jars = new ArrayList<String>();
        if (root == null || !Files.isDirectory(root)) return jars;

        Files.walkFileTree(root, new SimpleFileVisitor<Path>() {
            @Override public FileVisitResult visitFile(Path file, java.nio.file.attribute.BasicFileAttributes attrs) {
                if (file.toString().endsWith(".jar")) {
                    jars.add(file.toAbsolutePath().toString());
                }
                return FileVisitResult.CONTINUE;
            }
        });
        return jars;
    }

    private static int countJarsInClasspath(String cp) {
        int n = 0;
        String[] parts = cp.split(File.pathSeparator);
        for (int i = 0; i < parts.length; i++) {
            if (parts[i].endsWith(".jar")) n++;
        }
        return n;
    }

    /** 扫描 classesDir，将 outerFqns 对应的内部类（Outer$*.class）加入白名单 */
    private static Set<String> expandWithInnerClasses(String classesDir, List<String> outerFqns) throws IOException {
        Set<String> result = new LinkedHashSet<String>(outerFqns);
        for (String fqn : outerFqns) {
            Path outerClass = Paths.get(classesDir, fqn.replace('.', '/') + ".class");
            Path dir = outerClass.getParent();
            if (dir == null || !Files.isDirectory(dir)) continue;

            String simple = fqn.substring(fqn.lastIndexOf('.') + 1); // SimpleName
            DirectoryStream<Path> ds = null;
            try {
                ds = Files.newDirectoryStream(dir, simple + "$*.class");
                for (Path p : ds) {
                    String file = p.getFileName().toString();          // Simple$Inner.class
                    String innerSimple = file.substring(0, file.length() - ".class".length());
                    String pkg = "";
                    int i = fqn.lastIndexOf('.');
                    if (i > 0) pkg = fqn.substring(0, i + 1);
                    String innerFqn = pkg + innerSimple;               // 包 + Simple$Inner
                    result.add(innerFqn);
                }
            } finally {
                if (ds != null) ds.close();
            }
        }
        return result;
    }

    private static String pkgPrefixOf(String fqn) {
        int i = fqn.lastIndexOf('.');
        return (i > 0) ? fqn.substring(0, i) : "";
    }

    private static void copyLoggerFiles(String projectPath, String srcClass, String srcJava) throws IOException {
        // Path classDir  = Paths.get(projectPath, "target/classes/com/example");
        Path classDir  = Paths.get(projectPath, "build/classes/com/example");
        Files.createDirectories(classDir);
        Files.copy(Paths.get(srcClass),  classDir.resolve("DynamicLogger.class"), StandardCopyOption.REPLACE_EXISTING);

        // 建议不要把 .java 也拷进工程（可能触发重复编译），如确需：
        // Path sourceDir = Paths.get(projectPath, "src/main/java/com/example");
        // Files.createDirectories(sourceDir);
        // Files.copy(Paths.get(srcJava),   sourceDir.resolve("DynamicLogger.java"), StandardCopyOption.REPLACE_EXISTING);
    }
}
