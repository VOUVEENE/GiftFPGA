#!/bin/bash

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 函数：显示使用帮助
show_usage() {
    echo -e "${BLUE}Usage:${NC}"
    echo "  $0 <json_file1> [json_file2] [json_file3] ..."
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  $0 test/FPGA03.json"
    echo "  $0 test/FPGA03.json test/FPGA03_no_gift.json"
    echo "  $0 test/*.json"
    echo ""
    echo -e "${BLUE}Output directories:${NC}"
    echo "  • FPGA03.json → results/FPGA03/FPGA03.log"
    echo "  • FPGA03_no_gift.json → results_no_gift/FPGA03/FPGA03_no_gift.log"
}

# 函数：检查和创建运行脚本
setup_runner() {
    local RUNNER_SCRIPT="run_gift.py"
    
    if [ ! -f "$RUNNER_SCRIPT" ]; then
        echo -e "${YELLOW}Creating GiFt runner script...${NC}"
        cat > "$RUNNER_SCRIPT" << 'EOF'
#!/usr/bin/env python3
import sys
import logging

# 关键：设置正确的日志级别，这样GiFt的日志才能显示
logging.basicConfig(level=logging.INFO, format='[%(levelname)-7s] %(name)s - %(message)s', stream=sys.stdout)

# 添加模块路径
sys.path.insert(0, 'dreamplacefpga')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_gift.py <json_file>")
        sys.exit(1)
    
    from Placer import placeFPGA
    from Params import ParamsFPGA
    
    json_file = sys.argv[1]
    params = ParamsFPGA()
    params.load(json_file)
    placeFPGA(params)
EOF
        chmod +x "$RUNNER_SCRIPT"
        echo -e "${GREEN}✓ Created $RUNNER_SCRIPT${NC}"
    fi
}

# 函数：处理单个JSON文件
process_json() {
    local JSON_FILE=$1
    
    # 检查文件是否存在
    if [ ! -f "$JSON_FILE" ]; then
        echo -e "${RED}Error: File '$JSON_FILE' not found!${NC}"
        return 1
    fi
    
    local JSON_BASENAME=$(basename "$JSON_FILE" .json)
    
    # 判断是否包含 _no_gift
    if [[ "$JSON_BASENAME" == *"_no_gift"* ]]; then
        # 提取基础名称 (去掉_no_gift)
        local BASE_NAME=${JSON_BASENAME/_no_gift/}
        local RESULT_DIR="results_no_gift/$BASE_NAME"
        local LOG_FILE="$RESULT_DIR/${JSON_BASENAME}.log"
    else
        local RESULT_DIR="results/$JSON_BASENAME"
        local LOG_FILE="$RESULT_DIR/${JSON_BASENAME}.log"
    fi
    
    # 创建目录
    mkdir -p "$RESULT_DIR"
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${YELLOW}Processing: $JSON_FILE${NC}"
    echo -e "${YELLOW}Log file: $LOG_FILE${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # 记录开始时间
    local START_TIME=$(date +%s)
    
    # 使用新的运行方式 - 重要修改在这里！
    if python run_gift.py "$JSON_FILE" 2>&1 | tee "$LOG_FILE"; then
        local END_TIME=$(date +%s)
        local DURATION=$((END_TIME - START_TIME))
        echo -e "${GREEN}✓ Successfully completed: $JSON_FILE${NC}"
        echo -e "${GREEN}  Duration: ${DURATION}s${NC}"
        echo -e "${GREEN}  Log saved to: $LOG_FILE${NC}"
        
        # 提取关键信息
        if grep -q "GiFt优化有效" "$LOG_FILE"; then
            local GIFT_IMPROVEMENT=$(grep "GiFt优化有效" "$LOG_FILE" | grep -o '[0-9.]*%')
            echo -e "${GREEN}  GiFt improvement: ${GIFT_IMPROVEMENT}${NC}"
        fi
        
        # 提取最终HPWL
        if grep -q "Placement completed" "$LOG_FILE"; then
            local FINAL_HPWL=$(grep -B 1 "Placement completed" "$LOG_FILE" | grep "HPWL" | tail -1 | grep -o '[0-9.]*E[+-][0-9]*')
            echo -e "${GREEN}  Final HPWL: ${FINAL_HPWL}${NC}"
        fi
        
        return 0
    else
        echo -e "${RED}✗ Failed to process: $JSON_FILE${NC}"
        echo -e "${RED}  Check log file: $LOG_FILE${NC}"
        return 1
    fi
}

# 主程序开始
echo -e "${BLUE}DREAMPlace FPGA Placement Runner (with GiFt support)${NC}"
echo -e "${BLUE}===================================================${NC}"

# 设置运行环境
setup_runner

# 检查参数
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No JSON files specified!${NC}"
    echo ""
    show_usage
    exit 1
fi

# 统计变量
TOTAL_FILES=$#
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_FILES=()

# 记录总开始时间
TOTAL_START_TIME=$(date +%s)

echo -e "${YELLOW}Total files to process: $TOTAL_FILES${NC}"
echo ""

# 处理每个JSON文件
for JSON_FILE in "$@"; do
    if process_json "$JSON_FILE"; then
        ((SUCCESS_COUNT++))
    else
        ((FAILED_COUNT++))
        FAILED_FILES+=("$JSON_FILE")
    fi
    echo ""
done

# 计算总时间
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))

# 显示最终统计
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Final Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Total files: $TOTAL_FILES${NC}"
echo -e "${GREEN}Successful: $SUCCESS_COUNT${NC}"
echo -e "${RED}Failed: $FAILED_COUNT${NC}"
echo -e "${YELLOW}Total time: ${TOTAL_DURATION}s${NC}"

# 如果有失败的文件，列出来
if [ $FAILED_COUNT -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed files:${NC}"
    for failed_file in "${FAILED_FILES[@]}"; do
        echo -e "${RED}  • $failed_file${NC}"
    done
fi

echo -e "${BLUE}========================================${NC}"

# 退出码：如果所有文件都成功则返回0，否则返回1
if [ $FAILED_COUNT -eq 0 ]; then
    echo -e "${GREEN}All files processed successfully!${NC}"
    exit 0
else
    echo -e "${RED}Some files failed to process!${NC}"
    exit 1
fi