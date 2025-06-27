#!/usr/bin/env python3
"""
Command Line Interface for LLM Cooperation System
"""
import asyncio
import click
import json
import sys
from pathlib import Path
from typing import Optional

from . import __version__, get_info
from .config import SystemConfig, load_environment_config
from .engines.openai_engine import OpenAIEngine
from .managers.model_manager import ModelResourceManager
from .routing.intelligent_router import IntelligentRouter

@click.group()
@click.version_option(__version__)
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """LLM Cooperation System CLI"""
    ctx.ensure_object(dict)
    ctx.obj['config_file'] = config
    ctx.obj['verbose'] = verbose
    
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

@cli.command()
def info():
    """Show package information"""
    info_data = get_info()
    click.echo(f"LLM Cooperation System v{info_data['version']}")
    click.echo(f"Author: {info_data['author']}")
    click.echo(f"Description: {info_data['description']}")

@cli.command()
@click.option('--query', '-q', required=True, help='Query to process')
@click.option('--model', '-m', help='Specific model to use')
@click.option('--strategy', '-s', help='Routing strategy')
@click.option('--output', '-o', help='Output file')
@click.pass_context
def query(ctx, query, model, strategy, output):
    """Process a single query"""
    async def _query():
        try:
            # Initialize system
            config = SystemConfig()
            engine = OpenAIEngine(config)
            router = IntelligentRouter()
            
            await engine.initialize()
            
            # Process query
            preferences = {}
            if strategy:
                preferences['strategy'] = strategy
            
            if model:
                # Direct model inference
                from .engines.base_engine import InferenceRequest
                request = InferenceRequest(prompt=query, model_name=model)
                result = await engine.inference(request)
                response_text = result.text
            else:
                # Intelligent routing
                response_text = await router.route_request(query, preferences)
            
            # Output result
            if output:
                with open(output, 'w') as f:
                    f.write(response_text)
                click.echo(f"Result saved to {output}")
            else:
                click.echo(response_text)
                
            await engine.shutdown()
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_query())

@cli.command()
@click.option('--format', '-f', type=click.Choice(['json', 'table']), default='table')
@click.pass_context
def models(ctx, format):
    """List available models"""
    async def _models():
        try:
            config = SystemConfig()
            engine = OpenAIEngine(config)
            await engine.initialize()
            
            models_list = await engine.list_available_models()
            
            if format == 'json':
                click.echo(json.dumps(models_list, indent=2))
            else:
                click.echo("Available Models:")
                click.echo("-" * 60)
                for model in models_list:
                    status_color = 'green' if model['status'] == 'healthy' else 'red'
                    click.echo(f"📋 {model['name']}")
                    click.echo(f"   Path: {model['model_path']}")
                    click.echo(f"   Status: ", nl=False)
                    click.secho(model['status'], fg=status_color)
                    click.echo(f"   Tasks: {', '.join(model['supported_tasks'])}")
                    click.echo()
            
            await engine.shutdown()
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_models())

@cli.command()
@click.pass_context  
def status(ctx):
    """Show system status"""
    async def _status():
        try:
            config = SystemConfig()
            engine = OpenAIEngine(config)
            await engine.initialize()
            
            metrics = await engine.get_system_metrics()
            
            click.echo("🚀 LLM Cooperation System Status")
            click.echo("=" * 40)
            click.echo(f"Total Models: {metrics['total_models']}")
            click.echo(f"Available Models: {metrics['available_models']}")
            click.echo(f"Healthy Models: {metrics['healthy_models']}")
            
            health_pct = metrics['health_percentage']
            if health_pct >= 80:
                color = 'green'
                status_emoji = "✅"
            elif health_pct >= 50:
                color = 'yellow'
                status_emoji = "⚠️"
            else:
                color = 'red'
                status_emoji = "❌"
            
            click.echo(f"System Health: {status_emoji} ", nl=False)
            click.secho(f"{health_pct:.1f}%", fg=color)
            
            click.echo(f"\nAPI Endpoints:")
            for endpoint in metrics['api_endpoints']:
                click.echo(f"  • {endpoint}")
            
            await engine.shutdown()
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_status())

@cli.command()
@click.option('--model', '-m', help='Test specific model')
@click.pass_context
def test(ctx, model):
    """Test model connectivity"""
    async def _test():
        try:
            config = SystemConfig()
            engine = OpenAIEngine(config)
            await engine.initialize()
            
            if model:
                models_to_test = [model]
            else:
                models_to_test = list(config.MODELS.keys())
            
            click.echo("🧪 Testing Model Connectivity")
            click.echo("=" * 40)
            
            for model_name in models_to_test:
                click.echo(f"Testing {model_name}... ", nl=False)
                
                try:
                    test_result = await engine._test_model_connectivity(model_name)
                    if test_result:
                        click.secho("✅ PASS", fg='green')
                    else:
                        click.secho("❌ FAIL", fg='red')
                except Exception as e:
                    click.secho(f"❌ ERROR: {e}", fg='red')
            
            await engine.shutdown()
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_test())

@cli.command()
@click.option('--base-url', required=True, help='API base URL')
@click.option('--api-key', required=True, help='API key')
@click.pass_context
def configure(ctx, base_url, api_key):
    """Configure API settings"""
    try:
        config = SystemConfig()
        config.update_api_config(base_url, api_key)
        
        # Save to environment file
        env_content = f"BASE_URL={base_url}\nAPI_KEY={api_key}\n"
        with open('.env', 'w') as f:
            f.write(env_content)
        
        click.echo("✅ Configuration updated successfully")
        click.echo(f"Base URL: {base_url}")
        click.echo("API Key: [HIDDEN]")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--host', default='0.0.0.0', help='Server host')
@click.option('--port', default=8080, help='Server port') 
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@click.pass_context
def serve(ctx, host, port, reload):
    """Start the LLM Cooperation server"""
    try:
        import uvicorn
        from .server.main import create_app
        
        app = create_app()
        
        click.echo(f"🚀 Starting LLM Cooperation Server on {host}:{port}")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
        
    except ImportError:
        click.echo("Server dependencies not installed. Install with: pip install llm-cooperation[server]", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--query', '-q', required=True, help='Query for cooperation')
@click.option('--mode', '-m', type=click.Choice(['sequential', 'parallel', 'voting']), 
              default='sequential', help='Cooperation mode')
@click.option('--models', help='Comma-separated list of models')
@click.pass_context
def cooperate(ctx, query, mode, models):
    """Test multi-model cooperation"""
    async def _cooperate():
        try:
            config = SystemConfig()
            engine = OpenAIEngine(config)
            await engine.initialize()
            
            from .schedulers.cooperation_scheduler import CooperationScheduler
            scheduler = CooperationScheduler()
            
            if models:
                model_list = [m.strip() for m in models.split(',')]
            else:
                model_list = ['qwen3_32b', 'qwen3_8b']
            
            click.echo(f"🤝 Running {mode} cooperation with models: {model_list}")
            click.echo(f"Query: {query}")
            click.echo("-" * 60)
            
            if mode == 'sequential':
                result = await scheduler.create_sequential_task(query, model_list)
            elif mode == 'parallel':
                result = await scheduler.create_parallel_task(query, model_list)
            elif mode == 'voting':
                result = await scheduler.create_voting_task(query, model_list)
            
            click.echo("Result:")
            click.echo(result)
            
            await engine.shutdown()
            
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_cooperate())

def main():
    """Main CLI entry point"""
    cli()

if __name__ == '__main__':
    main()