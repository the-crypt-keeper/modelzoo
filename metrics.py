from typing import List, Dict, Optional
import re
from dataclasses import dataclass

@dataclass
class ModelMetrics:
    """Data class representing standardized model metrics."""
    num_running: Optional[int] = None
    num_waiting: Optional[int] = None
    kv_utilization: Optional[float] = None

    def __str__(self) -> str:
        return f"ModelMetrics(running={self.num_running}, waiting={self.num_waiting}, kv_util={self.kv_utilization})"

def parse_prometheus_metrics(text: str) -> List[Dict]:
    """Parse prometheus format metrics into structured data.
    
    Args:
        text (str): Prometheus format metrics text
        
    Returns:
        List[Dict]: List of metrics with keys: key, tags, value
    """
    metrics = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Parse metric line: metric_name{tag1="value1",tag2="value2"} value
        match = re.match(r'^([^{]+)(?:\{([^}]*)\})?\s+(.+)$', line)
        if not match:
            continue
            
        metric_name = match.group(1)
        tags_str = match.group(2) or ""
        value_str = match.group(3)
        
        # Parse tags
        tags = {}
        if tags_str:
            tag_matches = re.findall(r'(\w+)="([^"]*)"', tags_str)
            tags = dict(tag_matches)
        
        # Parse value
        try:
            value = float(value_str)
        except ValueError:
            continue
            
        metrics.append({
            'key': metric_name,
            'tags': tags,
            'value': value
        })
    
    return metrics

def vllm_metrics(text: str) -> ModelMetrics:
    """Extract standardized model metrics from vLLM prometheus metrics.
    
    Args:
        text (str): Raw prometheus format metrics text
        
    Returns:
        ModelMetrics: Standardized metrics object
    """
    prometheus_metrics = parse_prometheus_metrics(text)
    metrics = ModelMetrics()
    
    for metric in prometheus_metrics:
        key = metric['key']
        value = metric['value']
        
        if key == 'vllm:num_requests_running':
            metrics.num_running = int(value)
        elif key == 'vllm:num_requests_waiting':
            metrics.num_waiting = int(value)
        elif key == 'vllm:kv_cache_usage_perc':
            metrics.kv_utilization = value
    
    return metrics
