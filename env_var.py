# 设置变量

def to_value(s):
	"""
	变量转换
	:param s: 输入字符串
	:return: 根据字符串特征转换为对应类型
	"""
	try:
		return int(s)
	except:
		try:
			return float(s)
		except:
			if s.strip().lower() == "true":
				return True
			elif s.strip().lower() == "false":
				return False
			else:
				return s


class EnvVar:
	
	def __init__(self, config_file):
		self.config = {}
	
		with open(config_file, "r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line or "=" not in line or line[0] == "#":
					continue
				
				key, val = line.split("=", 1)
				self.config[key.strip()] = to_value(val.strip())
	

	def get_val(self, key):
		"""
		根据键返回值
		:param key: 键
		:return: 值
		"""
		return self.config[key]


if __name__ == "__main__":
	
	evar = EnvVar("env_var.txt")
	print(evar.get_val("time"), type(evar.get_val("time")))