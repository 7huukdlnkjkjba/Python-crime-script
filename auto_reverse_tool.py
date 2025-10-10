#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全自动逆向软件修改工具
功能包括：自动分析、识别和修改目标软件，减少手动干预
"""

import os
import sys
import struct
import argparse
import hashlib
import logging
import json
import re
import time
import subprocess
from typing import Dict, List, Tuple, Optional, Union, BinaryIO, Any
from dataclasses import dataclass, asdict
from enum import Enum

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AutoReverseTool')

class FileFormat(Enum):
    """文件格式枚举"""
    UNKNOWN = 0
    PE = 1      # Windows可执行文件
    ELF = 2     # Linux可执行文件
    MACHO = 3   # macOS可执行文件

class PatchType(Enum):
    """修补类型枚举"""
    NOP = 1             # NOP指令替换
    JMP = 2             # 跳转指令
    CALL = 3            # 调用指令
    RET = 4             # 返回指令
    BYTES = 5           # 字节替换
    STRING = 6          # 字符串替换
    LICENSE_CHECK = 7   # 许可证检查绕过
    TIME_BOMB = 8       # 时间炸弹移除

@dataclass
class PatchInfo:
    """修补信息数据类"""
    offset: int
    original_bytes: bytes
    new_bytes: bytes
    patch_type: PatchType
    description: str = ""
    confidence: float = 1.0  # 置信度 (0.0-1.0)

@dataclass
class AnalysisResult:
    """分析结果数据类"""
    file_path: str
    file_format: FileFormat
    file_size: int
    md5: str
    sha1: str
    sha256: str
    sections: List[Dict[str, Any]]
    imports: List[Dict[str, Any]]
    exports: List[Dict[str, Any]]
    strings: List[str]
    suspicious_patterns: List[Dict[str, Any]]
    potential_patches: List[PatchInfo]
    analysis_time: float

class AutoReverseTool:
    """全自动逆向软件修改工具类"""
    
    def __init__(self, file_path: str, config_path: Optional[str] = None):
        """
        初始化自动逆向工具
        
        Args:
            file_path: 目标文件路径
            config_path: 配置文件路径
        """
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.file_size = os.path.getsize(file_path)
        self.file_data = None
        self.file_format = FileFormat.UNKNOWN
        self.sections = []
        self.imports = []
        self.exports = []
        self.strings = []
        self.suspicious_patterns = []
        self.potential_patches = []
        self.config = self._load_config(config_path)
        
        # 加载文件数据
        self._load_file()
        
        # 分析文件类型
        self._analyze_file_type()
        
        # 分析文件结构
        self._analyze_file_structure()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        default_config = {
            "string_min_length": 4,
            "suspicious_strings": [
                "license", "key", "serial", "activation", "trial", "expire",
                "register", "unregistered", "demo", "evaluation", "crack",
                "protection", "anti-debug", "anti-tamper", "check"
            ],
            "suspicious_apis": [
                "IsDebuggerPresent", "CheckRemoteDebuggerPresent", "OutputDebugString",
                "GetTickCount", "QueryPerformanceCounter", "timeGetTime",
                "GetLocalTime", "GetSystemTime", "GetFileTime",
                "CreateFile", "ReadFile", "WriteFile", "DeleteFile",
                "RegOpenKey", "RegQueryValue", "RegSetValue", "RegDeleteKey",
                "CreateMutex", "OpenMutex", "CreateEvent", "OpenEvent",
                "VirtualProtect", "VirtualAlloc", "VirtualFree",
                "GetModuleHandle", "GetProcAddress", "LoadLibrary"
            ],
            "patch_patterns": {
                "license_check": [
                    {"pattern": "85 C0 74 ??", "description": "test eax, eax; jz near", "confidence": 0.8},
                    {"pattern": "83 F8 00 74 ??", "description": "cmp eax, 0; jz near", "confidence": 0.8},
                    {"pattern": "83 7D ?? 00 74 ??", "description": "cmp [ebp+??], 0; jz near", "confidence": 0.7}
                ],
                "time_bomb": [
                    {"pattern": "FF 15 ?? ?? ?? ?? 85 C0 74 ??", "description": "call dword ptr []; test eax, eax; jz near", "confidence": 0.7},
                    {"pattern": "E8 ?? ?? ?? ?? 85 C0 74 ??", "description": "call near; test eax, eax; jz near", "confidence": 0.7}
                ],
                "anti_debug": [
                    {"pattern": "64 A1 ?? ?? ?? ?? ?? 85 C0 74 ??", "description": "mov eax, fs:[0]; test eax, eax; jz near", "confidence": 0.9}
                ]
            },
            "auto_patch": True,
            "backup_original": True,
            "output_dir": "patched_files"
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 合并用户配置和默认配置
                    for key, value in user_config.items():
                        if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
                logger.info(f"成功加载配置文件: {config_path}")
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
        
        return default_config
    
    def _load_file(self):
        """加载文件数据到内存"""
        try:
            with open(self.file_path, 'rb') as f:
                self.file_data = f.read()
            logger.info(f"成功加载文件: {self.file_path} (大小: {self.file_size} 字节)")
        except Exception as e:
            logger.error(f"加载文件失败: {e}")
            raise
    
    def _analyze_file_type(self):
        """分析文件类型"""
        if not self.file_data:
            logger.error("文件数据未加载")
            return
        
        # 检查MZ头 (DOS/PE文件)
        if self.file_data[:2] == b'MZ':
            self.file_format = FileFormat.PE
            logger.info("检测到PE文件格式")
            self._parse_pe_header()
        # 检查ELF头 (Linux可执行文件)
        elif self.file_data[:4] == b'\x7fELF':
            self.file_format = FileFormat.ELF
            logger.info("检测到ELF文件格式")
            self._parse_elf_header()
        # 检查Mach-O头 (macOS可执行文件)
        elif self.file_data[:4] in [b'\xfe\xed\xfa\xce', b'\xfe\xed\xfa\xcf', 
                                    b'\xce\xfa\xed\xfe', b'\xcf\xfa\xed\xfe']:
            self.file_format = FileFormat.MACHO
            logger.info("检测到Mach-O文件格式")
            self._parse_macho_header()
        else:
            self.file_format = FileFormat.UNKNOWN
            logger.info("未知文件格式")
    
    def _parse_pe_header(self):
        """解析PE文件头"""
        try:
            # DOS头
            dos_header = self.file_data[:64]
            e_lfanew = struct.unpack('<I', dos_header[60:64])[0]
            
            # PE签名
            pe_signature = self.file_data[e_lfanew:e_lfanew+4]
            if pe_signature != b'PE\0\0':
                logger.error("无效的PE签名")
                return
            
            # COFF头
            coff_header = self.file_data[e_lfanew+4:e_lfanew+20]
            machine, num_sections, timestamp, sym_table, num_symbols, opt_header_size, characteristics = \
                struct.unpack('<HHIIIHH', coff_header)
            
            # 可选头
            opt_header_offset = e_lfanew + 20
            magic = self.file_data[opt_header_offset:opt_header_offset+2]
            
            if magic == b'\x0b\x01':  # PE32
                logger.info("PE32文件")
                # 解析PE32可选头
                opt_header = self.file_data[opt_header_offset:opt_header_offset+opt_header_size]
                entry_point = struct.unpack('<I', opt_header[16:20])[0]
                image_base = struct.unpack('<I', opt_header[28:32])[0]
                
                # 解析数据目录
                data_dirs = []
                data_dir_offset = opt_header_offset + 92  # PE32可选头中的数据目录偏移
                for i in range(16):  # PE有16个数据目录
                    va, size = struct.unpack('<II', self.file_data[data_dir_offset+i*8:data_dir_offset+(i+1)*8])
                    data_dirs.append({'va': va, 'size': size})
                
                # 解析导入表
                if data_dirs[1]['va'] and data_dirs[1]['size']:
                    self._parse_import_table(data_dirs[1]['va'], data_dirs[1]['size'], image_base)
                
                # 解析导出表
                if data_dirs[0]['va'] and data_dirs[0]['size']:
                    self._parse_export_table(data_dirs[0]['va'], data_dirs[0]['size'], image_base)
                
            elif magic == b'\x0b\x02':  # PE32+
                logger.info("PE32+文件")
                # 解析PE32+可选头
                opt_header = self.file_data[opt_header_offset:opt_header_offset+opt_header_size]
                entry_point = struct.unpack('<I', opt_header[16:20])[0]
                image_base = struct.unpack('<Q', opt_header[24:32])[0]
                
                # 解析数据目录
                data_dirs = []
                data_dir_offset = opt_header_offset + 108  # PE32+可选头中的数据目录偏移
                for i in range(16):  # PE有16个数据目录
                    va, size = struct.unpack('<II', self.file_data[data_dir_offset+i*8:data_dir_offset+(i+1)*8])
                    data_dirs.append({'va': va, 'size': size})
                
                # 解析导入表
                if data_dirs[1]['va'] and data_dirs[1]['size']:
                    self._parse_import_table(data_dirs[1]['va'], data_dirs[1]['size'], image_base)
                
                # 解析导出表
                if data_dirs[0]['va'] and data_dirs[0]['size']:
                    self._parse_export_table(data_dirs[0]['va'], data_dirs[0]['size'], image_base)
            
            # 节表
            sections_offset = opt_header_offset + opt_header_size
            section_size = 40  # 每个节表项的大小
            
            for i in range(num_sections):
                section_data = self.file_data[sections_offset + i*section_size : sections_offset + (i+1)*section_size]
                name = section_data[:8].strip(b'\x00').decode('ascii', errors='ignore')
                virtual_size, virtual_address, raw_data_size, raw_data_ptr, reloc_ptr, line_num_ptr, \
                num_relocs, num_line_nums, characteristics = struct.unpack('<IIIIIIHHI', section_data[8:40])
                
                self.sections.append({
                    'name': name,
                    'virtual_size': virtual_size,
                    'virtual_address': virtual_address,
                    'raw_data_size': raw_data_size,
                    'raw_data_ptr': raw_data_ptr,
                    'characteristics': characteristics
                })
            
            logger.info(f"解析了 {len(self.sections)} 个节")
            
        except Exception as e:
            logger.error(f"解析PE头失败: {e}")
    
    def _parse_import_table(self, va: int, size: int, image_base: int):
        """解析导入表"""
        try:
            # 将VA转换为文件偏移
            offset = self._va_to_offset(va)
            if offset is None:
                return
            
            # 遍历导入描述符
            import_desc_size = 20  # 每个导入描述符的大小
            for i in range(size // import_desc_size):
                desc_offset = offset + i * import_desc_size
                if desc_offset + import_desc_size > len(self.file_data):
                    break
                
                # 读取导入描述符
                original_thunk_rva, timestamp, forwarder_chain, name_rva, first_thunk_rva = \
                    struct.unpack('<IIIII', self.file_data[desc_offset:desc_offset+20])
                
                # 检查是否为空描述符
                if original_thunk_rva == 0 and name_rva == 0 and first_thunk_rva == 0:
                    break
                
                # 获取DLL名称
                if name_rva:
                    name_offset = self._va_to_offset(name_rva)
                    if name_offset:
                        dll_name = ""
                        j = 0
                        while name_offset + j < len(self.file_data) and self.file_data[name_offset + j] != 0:
                            dll_name += chr(self.file_data[name_offset + j])
                            j += 1
                        
                        if dll_name:
                            # 获取导入函数
                            functions = []
                            thunk_offset = self._va_to_offset(first_thunk_rva)
                            if thunk_offset:
                                while thunk_offset + 4 <= len(self.file_data):
                                    thunk = struct.unpack('<I', self.file_data[thunk_offset:thunk_offset+4])[0]
                                    if thunk == 0:
                                        break
                                    
                                    # 检查是按序号导入还是按名称导入
                                    if thunk & 0x80000000:  # 按序号导入
                                        ordinal = thunk & 0xFFFF
                                        functions.append({
                                            'type': 'ordinal',
                                            'value': ordinal,
                                            'name': f"Ordinal_{ordinal}"
                                        })
                                    else:  # 按名称导入
                                        name_offset = self._va_to_offset(thunk)
                                        if name_offset:
                                            func_name = ""
                                            hint = struct.unpack('<H', self.file_data[name_offset:name_offset+2])[0]
                                            j = 2
                                            while name_offset + j < len(self.file_data) and self.file_data[name_offset + j] != 0:
                                                func_name += chr(self.file_data[name_offset + j])
                                                j += 1
                                            
                                            functions.append({
                                                'type': 'name',
                                                'value': hint,
                                                'name': func_name
                                            })
                                    
                                    thunk_offset += 4
                            
                            self.imports.append({
                                'dll': dll_name,
                                'functions': functions
                            })
            
            logger.info(f"解析了 {len(self.imports)} 个导入DLL")
            
        except Exception as e:
            logger.error(f"解析导入表失败: {e}")
    
    def _parse_export_table(self, va: int, size: int, image_base: int):
        """解析导出表"""
        try:
            # 将VA转换为文件偏移
            offset = self._va_to_offset(va)
            if offset is None:
                return
            
            # 读取导出目录
            export_flags, timestamp, major_version, minor_version, name_rva, ordinal_base, \
            num_functions, num_names, address_table_rva, name_table_rva, ordinal_table_rva = \
                struct.unpack('<IIHHIIIIII', self.file_data[offset:offset+40])
            
            # 获取DLL名称
            if name_rva:
                name_offset = self._va_to_offset(name_rva)
                if name_offset:
                    dll_name = ""
                    j = 0
                    while name_offset + j < len(self.file_data) and self.file_data[name_offset + j] != 0:
                        dll_name += chr(self.file_data[name_offset + j])
                        j += 1
            
            # 解析导出函数
            address_table_offset = self._va_to_offset(address_table_rva)
            name_table_offset = self._va_to_offset(name_table_rva)
            ordinal_table_offset = self._va_to_offset(ordinal_table_rva)
            
            if address_table_offset and name_table_offset and ordinal_table_offset:
                for i in range(num_names):
                    # 读取函数名RVA
                    name_rva = struct.unpack('<I', self.file_data[name_table_offset+i*4:name_table_offset+i*4+4])[0]
                    name_offset = self._va_to_offset(name_rva)
                    if name_offset:
                        func_name = ""
                        j = 0
                        while name_offset + j < len(self.file_data) and self.file_data[name_offset + j] != 0:
                            func_name += chr(self.file_data[name_offset + j])
                            j += 1
                        
                        # 读取序号
                        ordinal = struct.unpack('<H', self.file_data[ordinal_table_offset+i*2:ordinal_table_offset+i*2+2])[0]
                        
                        # 读取函数地址
                        func_rva = struct.unpack('<I', self.file_data[address_table_offset+ordinal*4:address_table_offset+ordinal*4+4])[0]
                        
                        self.exports.append({
                            'name': func_name,
                            'ordinal': ordinal + ordinal_base,
                            'rva': func_rva
                        })
            
            logger.info(f"解析了 {len(self.exports)} 个导出函数")
            
        except Exception as e:
            logger.error(f"解析导出表失败: {e}")
    
    def _va_to_offset(self, va: int) -> Optional[int]:
        """
        将虚拟地址(VA)转换为文件偏移
        
        Args:
            va: 虚拟地址
            
        Returns:
            文件偏移，如果转换失败则返回None
        """
        if not self.sections:
            return None
        
        for section in self.sections:
            section_va = section['virtual_address']
            section_size = section['virtual_size']
            section_offset = section['raw_data_ptr']
            
            if section_va <= va < section_va + section_size:
                return section_offset + (va - section_va)
        
        return None
    
    def _parse_elf_header(self):
        """解析ELF文件头"""
        # 简化的ELF头解析
        try:
            elf_class = self.file_data[4]  # 1=32位, 2=64位
            encoding = self.file_data[5]    # 1=小端, 2=大端
            
            if elf_class == 1:
                logger.info("32位ELF文件")
                # 32位ELF头解析
                # 这里可以添加32位ELF解析代码
            elif elf_class == 2:
                logger.info("64位ELF文件")
                # 64位ELF头解析
                # 这里可以添加64位ELF解析代码
            else:
                logger.error("无效的ELF类别")
        except Exception as e:
            logger.error(f"解析ELF头失败: {e}")
    
    def _parse_macho_header(self):
        """解析Mach-O文件头"""
        # 简化的Mach-O头解析
        try:
            magic = self.file_data[:4]
            if magic in [b'\xfe\xed\xfa\xce', b'\xce\xfa\xed\xfe']:
                logger.info("32位Mach-O文件")
                # 32位Mach-O头解析
                # 这里可以添加32位Mach-O解析代码
            elif magic in [b'\xfe\xed\xfa\xcf', b'\xcf\xfa\xed\xfe']:
                logger.info("64位Mach-O文件")
                # 64位Mach-O头解析
                # 这里可以添加64位Mach-O解析代码
        except Exception as e:
            logger.error(f"解析Mach-O头失败: {e}")
    
    def _analyze_file_structure(self):
        """分析文件结构"""
        # 提取字符串
        self.extract_strings(self.config["string_min_length"])
        
        # 检测可疑模式
        self._detect_suspicious_patterns()
        
        # 识别潜在的修补点
        self._identify_potential_patches()
    
    def extract_strings(self, min_length: int = 4) -> List[str]:
        """
        提取文件中的字符串
        
        Args:
            min_length: 字符串最小长度
            
        Returns:
            提取到的字符串列表
        """
        if not self.file_data:
            logger.error("文件数据未加载")
            return []
        
        strings = []
        current_string = ""
        
        for byte in self.file_data:
            # 检查是否为可打印ASCII字符
            if 32 <= byte <= 126:
                current_string += chr(byte)
            else:
                if len(current_string) >= min_length:
                    strings.append(current_string)
                current_string = ""
        
        # 检查最后一个字符串
        if len(current_string) >= min_length:
            strings.append(current_string)
        
        self.strings = strings
        logger.info(f"提取了 {len(strings)} 个字符串")
        return strings
    
    def _detect_suspicious_patterns(self):
        """检测可疑模式"""
        # 检测可疑字符串
        suspicious_strings = []
        for string in self.strings:
            for pattern in self.config["suspicious_strings"]:
                if pattern.lower() in string.lower():
                    suspicious_strings.append({
                        'type': 'string',
                        'pattern': pattern,
                        'value': string,
                        'description': f"发现可疑字符串: {string}"
                    })
        
        # 检测可疑API调用
        suspicious_apis = []
        for imp in self.imports:
            for func in imp['functions']:
                for api in self.config["suspicious_apis"]:
                    if api.lower() in func['name'].lower():
                        suspicious_apis.append({
                            'type': 'api',
                            'pattern': api,
                            'value': func['name'],
                            'dll': imp['dll'],
                            'description': f"发现可疑API调用: {imp['dll']}.{func['name']}"
                        })
        
        # 检测可疑字节模式
        suspicious_bytes = []
        for category, patterns in self.config["patch_patterns"].items():
            for pattern_info in patterns:
                # 将十六进制模式字符串转换为字节模式
                hex_pattern = pattern_info["pattern"].replace(' ', '')
                # 处理通配符
                regex_pattern = ""
                i = 0
                while i < len(hex_pattern):
                    if i + 1 < len(hex_pattern):
                        if hex_pattern[i:i+2] == "??":
                            regex_pattern += r"."
                            i += 2
                        else:
                            regex_pattern += hex_pattern[i:i+2]
                            i += 2
                    else:
                        i += 1
                
                # 创建正则表达式模式
                byte_pattern = bytes.fromhex(re.sub(r'(..)', r'\\x\1', regex_pattern))
                matches = self.search_bytes(byte_pattern)
                
                for match in matches:
                    suspicious_bytes.append({
                        'type': 'bytes',
                        'category': category,
                        'pattern': pattern_info["pattern"],
                        'offset': match,
                        'description': pattern_info["description"],
                        'confidence': pattern_info["confidence"]
                    })
        
        self.suspicious_patterns = suspicious_strings + suspicious_apis + suspicious_bytes
        logger.info(f"检测到 {len(self.suspicious_patterns)} 个可疑模式")
    
    def _identify_potential_patches(self):
        """识别潜在的修补点"""
        potential_patches = []
        
        # 基于可疑模式识别修补点
        for pattern in self.suspicious_patterns:
            if pattern['type'] == 'bytes':
                # 解析原始字节
                hex_pattern = pattern["pattern"].replace(' ', '')
                original_bytes = b''
                i = 0
                while i < len(hex_pattern):
                    if i + 1 < len(hex_pattern):
                        if hex_pattern[i:i+2] == "??":
                            # 通配符，使用原始文件中的字节
                            offset = pattern['offset'] + i // 2
                            if offset < len(self.file_data):
                                original_bytes += bytes([self.file_data[offset]])
                            i += 2
                        else:
                            original_bytes += bytes.fromhex(hex_pattern[i:i+2])
                            i += 2
                    else:
                        i += 1
                
                # 根据类别确定修补类型和修补字节
                patch_type = None
                new_bytes = None
                
                if pattern['category'] == 'license_check':
                    patch_type = PatchType.LICENSE_CHECK
                    # 将条件跳转改为无条件跳转或NOP
                    if original_bytes[-2:] == b'\x74':  # JZ
                        new_bytes = original_bytes[:-2] + b'\xEB'  # 改为JMP
                    elif original_bytes[-2:] == b'\x75':  # JNZ
                        new_bytes = original_bytes[:-2] + b'\xEB'  # 改为JMP
                    else:
                        # 其他情况，用NOP替换
                        new_bytes = b'\x90' * len(original_bytes)
                
                elif pattern['category'] == 'time_bomb':
                    patch_type = PatchType.TIME_BOMB
                    # 用NOP替换
                    new_bytes = b'\x90' * len(original_bytes)
                
                elif pattern['category'] == 'anti_debug':
                    patch_type = PatchType.NOP
                    # 用NOP替换
                    new_bytes = b'\x90' * len(original_bytes)
                
                if patch_type and new_bytes:
                    potential_patches.append(PatchInfo(
                        offset=pattern['offset'],
                        original_bytes=original_bytes,
                        new_bytes=new_bytes,
                        patch_type=patch_type,
                        description=pattern['description'],
                        confidence=pattern['confidence']
                    ))
        
        # 基于字符串识别修补点
        for string in self.strings:
            for pattern in self.config["suspicious_strings"]:
                if pattern.lower() in string.lower():
                    # 查找字符串在文件中的位置
                    string_bytes = string.encode('ascii')
                    matches = self.search_bytes(string_bytes)
                    
                    for match in matches:
                        # 创建修补信息，用空格替换字符串
                        new_bytes = b' ' * len(string_bytes)
                        potential_patches.append(PatchInfo(
                            offset=match,
                            original_bytes=string_bytes,
                            new_bytes=new_bytes,
                            patch_type=PatchType.STRING,
                            description=f"替换可疑字符串: {string}",
                            confidence=0.6
                        ))
        
        # 按置信度排序
        potential_patches.sort(key=lambda x: x.confidence, reverse=True)
        
        self.potential_patches = potential_patches
        logger.info(f"识别了 {len(potential_patches)} 个潜在修补点")
    
    def calculate_hashes(self) -> Dict[str, str]:
        """
        计算文件的哈希值
        
        Returns:
            包含各种哈希值的字典
        """
        if not self.file_data:
            logger.error("文件数据未加载")
            return {}
        
        hashes = {
            'md5': hashlib.md5(self.file_data).hexdigest(),
            'sha1': hashlib.sha1(self.file_data).hexdigest(),
            'sha256': hashlib.sha256(self.file_data).hexdigest()
        }
        
        logger.info(f"MD5: {hashes['md5']}")
        logger.info(f"SHA1: {hashes['sha1']}")
        logger.info(f"SHA256: {hashes['sha256']}")
        
        return hashes
    
    def patch_file(self, offset: int, data: bytes, output_path: Optional[str] = None) -> bool:
        """
        修补文件
        
        Args:
            offset: 修补的偏移量
            data: 要写入的数据
            output_path: 输出文件路径，如果为None则直接修改原文件
            
        Returns:
            修补是否成功
        """
        try:
            if offset < 0 or offset >= len(self.file_data):
                logger.error(f"无效的偏移量: {offset}")
                return False
            
            if offset + len(data) > len(self.file_data):
                logger.error("修补数据超出文件范围")
                return False
            
            # 创建修改后的数据
            modified_data = self.file_data[:offset] + data + self.file_data[offset+len(data):]
            
            # 确定输出路径
            if output_path is None:
                output_path = self.file_path
            
            # 写入文件
            with open(output_path, 'wb') as f:
                f.write(modified_data)
            
            logger.info(f"成功修补文件: {output_path} (偏移量: {offset}, 大小: {len(data)} 字节)")
            
            # 如果修改的是原文件，重新加载数据
            if output_path == self.file_path:
                self._load_file()
            
            return True
        except Exception as e:
            logger.error(f"修补文件失败: {e}")
            return False
    
    def search_bytes(self, pattern: bytes, start_offset: int = 0) -> List[int]:
        """
        在文件中搜索字节模式
        
        Args:
            pattern: 要搜索的字节模式
            start_offset: 开始搜索的偏移量
            
        Returns:
            匹配位置的列表
        """
        if not self.file_data:
            logger.error("文件数据未加载")
            return []
        
        if start_offset < 0 or start_offset >= len(self.file_data):
            logger.error(f"无效的起始偏移量: {start_offset}")
            return []
        
        matches = []
        offset = start_offset
        
        while offset < len(self.file_data):
            pos = self.file_data.find(pattern, offset)
            if pos == -1:
                break
            matches.append(pos)
            offset = pos + 1
        
        return matches
    
    def save_strings_to_file(self, output_path: str) -> bool:
        """
        将提取的字符串保存到文件
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            保存是否成功
        """
        try:
            if not self.strings:
                self.extract_strings()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for string in self.strings:
                    f.write(string + '\n')
            
            logger.info(f"成功保存字符串到文件: {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存字符串失败: {e}")
            return False
    
    def apply_auto_patches(self, output_path: Optional[str] = None, min_confidence: float = 0.7) -> bool:
        """
        自动应用修补
        
        Args:
            output_path: 输出文件路径，如果为None则使用默认路径
            min_confidence: 最小置信度阈值
            
        Returns:
            修补是否成功
        """
        if not self.config["auto_patch"]:
            logger.info("自动修补已禁用")
            return False
        
        # 确定输出路径
        if output_path is None:
            output_dir = self.config["output_dir"]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            base_name, ext = os.path.splitext(self.file_name)
            output_path = os.path.join(output_dir, f"{base_name}_patched{ext}")
        
        # 备份原始文件
        if self.config["backup_original"]:
            backup_path = output_path + ".bak"
            with open(self.file_path, 'rb') as src, open(backup_path, 'wb') as dst:
                dst.write(src.read())
            logger.info(f"原始文件已备份到: {backup_path}")
        
        # 复制原始文件到输出路径
        with open(self.file_path, 'rb') as src, open(output_path, 'wb') as dst:
            dst.write(src.read())
        
        # 应用高置信度的修补
        applied_patches = []
        for patch in self.potential_patches:
            if patch.confidence >= min_confidence:
                success = self._apply_patch_to_file(patch, output_path)
                if success:
                    applied_patches.append(patch)
                    logger.info(f"应用修补: {patch.description} (置信度: {patch.confidence})")
        
        logger.info(f"成功应用了 {len(applied_patches)} 个修补")
        return True
    
    def _apply_patch_to_file(self, patch: PatchInfo, file_path: str) -> bool:
        """
        将修补应用到文件
        
        Args:
            patch: 修补信息
            file_path: 文件路径
            
        Returns:
            修补是否成功
        """
        try:
            with open(file_path, 'r+b') as f:
                f.seek(patch.offset)
                f.write(patch.new_bytes)
            return True
        except Exception as e:
            logger.error(f"应用修补失败: {e}")
            return False
    
    def get_analysis_result(self) -> AnalysisResult:
        """
        获取分析结果
        
        Returns:
            分析结果对象
        """
        return AnalysisResult(
            file_path=self.file_path,
            file_format=self.file_format,
            file_size=self.file_size,
            md5=self.calculate_hashes()['md5'],
            sha1=self.calculate_hashes()['sha1'],
            sha256=self.calculate_hashes()['sha256'],
            sections=self.sections,
            imports=self.imports,
            exports=self.exports,
            strings=self.strings,
            suspicious_patterns=self.suspicious_patterns,
            potential_patches=self.potential_patches,
            analysis_time=time.time()
        )
    
    def save_analysis_result(self, output_path: str) -> bool:
        """
        保存分析结果到文件
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            保存是否成功
        """
        try:
            result = self.get_analysis_result()
            
            # 转换为可序列化的字典
            result_dict = asdict(result)
            result_dict['file_format'] = result.file_format.name
            result_dict['potential_patches'] = [
                {
                    'offset': p.offset,
                    'original_bytes': p.original_bytes.hex(),
                    'new_bytes': p.new_bytes.hex(),
                    'patch_type': p.patch_type.name,
                    'description': p.description,
                    'confidence': p.confidence
                }
                for p in result.potential_patches
            ]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"成功保存分析结果到文件: {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存分析结果失败: {e}")
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='全自动逆向软件修改工具')
    parser.add_argument('file', help='目标文件路径')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--auto-patch', action='store_true', help='自动应用修补')
    parser.add_argument('--output', help='输出文件路径')
    parser.add_argument('--min-confidence', type=float, default=0.7, help='最小置信度阈值 (0.0-1.0)')
    parser.add_argument('--save-analysis', help='保存分析结果到文件')
    parser.add_argument('--extract-strings', action='store_true', help='提取字符串')
    parser.add_argument('--strings-output', help='字符串输出文件路径')
    
    args = parser.parse_args()
    
    try:
        # 初始化工具
        tool = AutoReverseTool(args.file, args.config)
        
        # 显示文件信息
        info = tool.get_analysis_result()
        print("\n=== 文件信息 ===")
        print(f"文件路径: {info.file_path}")
        print(f"文件格式: {info.file_format.name}")
        print(f"文件大小: {info.file_size} 字节")
        print("\n=== 哈希值 ===")
        print(f"MD5: {info.md5}")
        print(f"SHA1: {info.sha1}")
        print(f"SHA256: {info.sha256}")
        
        if info.sections:
            print("\n=== 节信息 ===")
            for section in info.sections:
                print(f"名称: {section['name']}")
                print(f"  虚拟大小: {section['virtual_size']}")
                print(f"  虚拟地址: 0x{section['virtual_address']:08X}")
                print(f"  原始数据大小: {section['raw_data_size']}")
                print(f"  原始数据指针: 0x{section['raw_data_ptr']:08X}")
                print(f"  特征: 0x{section['characteristics']:08X}")
        
        if info.imports:
            print("\n=== 导入信息 ===")
            for imp in info.imports[:5]:  # 只显示前5个导入DLL
                print(f"DLL: {imp['dll']}")
                for func in imp['functions'][:3]:  # 每个DLL只显示前3个函数
                    print(f"  {func['name']}")
            if len(info.imports) > 5:
                print(f"... 还有 {len(info.imports) - 5} 个导入DLL")
        
        if info.suspicious_patterns:
            print("\n=== 可疑模式 ===")
            for pattern in info.suspicious_patterns[:10]:  # 只显示前10个可疑模式
                print(f"{pattern['description']} (置信度: {pattern.get('confidence', 'N/A')})")
            if len(info.suspicious_patterns) > 10:
                print(f"... 还有 {len(info.suspicious_patterns) - 10} 个可疑模式")
        
        if info.potential_patches:
            print("\n=== 潜在修补点 ===")
            for patch in info.potential_patches[:10]:  # 只显示前10个潜在修补点
                print(f"偏移量: 0x{patch.offset:08X}, 类型: {patch.patch_type.name}, 描述: {patch.description}, 置信度: {patch.confidence}")
            if len(info.potential_patches) > 10:
                print(f"... 还有 {len(info.potential_patches) - 10} 个潜在修补点")
        
        # 提取字符串
        if args.extract_strings:
            strings_output = args.strings_output or "extracted_strings.txt"
            if tool.save_strings_to_file(strings_output):
                print(f"\n字符串已保存到: {strings_output}")
        
        # 保存分析结果
        if args.save_analysis:
            if tool.save_analysis_result(args.save_analysis):
                print(f"\n分析结果已保存到: {args.save_analysis}")
        
        # 自动应用修补
        if args.auto_patch:
            success = tool.apply_auto_patches(args.output, args.min_confidence)
            if success:
                print(f"\n自动修补完成")
                if args.output:
                    print(f"修补后的文件: {args.output}")
                else:
                    output_dir = tool.config["output_dir"]
                    base_name, ext = os.path.splitext(tool.file_name)
                    output_path = os.path.join(output_dir, f"{base_name}_patched{ext}")
                    print(f"修补后的文件: {output_path}")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()