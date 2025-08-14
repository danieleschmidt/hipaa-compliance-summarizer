"""Internationalization (i18n) support for HIPAA compliance system."""

import json
import os
from pathlib import Path
from typing import Dict, Optional

# Supported languages with their locale codes
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Español', 
    'fr': 'Français',
    'de': 'Deutsch',
    'ja': '日本語',
    'zh': '中文'
}

# Default translations for core messages
DEFAULT_TRANSLATIONS = {
    'en': {
        'processing_document': 'Processing document',
        'document_processed': 'Document processed successfully',
        'phi_detected': 'PHI entities detected',
        'compliance_score': 'Compliance score',
        'error_processing': 'Error processing document',
        'security_validation_failed': 'Security validation failed',
        'invalid_file_path': 'Invalid file path',
        'file_too_large': 'File too large for processing',
        'batch_processing_complete': 'Batch processing complete',
        'health_check_passed': 'System health check passed',
        'cache_performance': 'Cache performance statistics'
    },
    'es': {
        'processing_document': 'Procesando documento',
        'document_processed': 'Documento procesado exitosamente',
        'phi_detected': 'Entidades PHI detectadas',
        'compliance_score': 'Puntuación de cumplimiento',
        'error_processing': 'Error procesando documento',
        'security_validation_failed': 'Falló la validación de seguridad',
        'invalid_file_path': 'Ruta de archivo inválida',
        'file_too_large': 'Archivo demasiado grande para procesar',
        'batch_processing_complete': 'Procesamiento por lotes completo',
        'health_check_passed': 'Verificación de salud del sistema pasó',
        'cache_performance': 'Estadísticas de rendimiento de caché'
    },
    'fr': {
        'processing_document': 'Traitement du document',
        'document_processed': 'Document traité avec succès',
        'phi_detected': 'Entités PHI détectées',
        'compliance_score': 'Score de conformité',
        'error_processing': 'Erreur lors du traitement du document',
        'security_validation_failed': 'Échec de la validation de sécurité',
        'invalid_file_path': 'Chemin de fichier invalide',
        'file_too_large': 'Fichier trop volumineux pour le traitement',
        'batch_processing_complete': 'Traitement par lots terminé',
        'health_check_passed': 'Vérification de santé système réussie',
        'cache_performance': 'Statistiques de performance du cache'
    },
    'de': {
        'processing_document': 'Dokument wird verarbeitet',
        'document_processed': 'Dokument erfolgreich verarbeitet',
        'phi_detected': 'PHI-Entitäten erkannt',
        'compliance_score': 'Compliance-Score',
        'error_processing': 'Fehler bei der Dokumentverarbeitung',
        'security_validation_failed': 'Sicherheitsvalidierung fehlgeschlagen',
        'invalid_file_path': 'Ungültiger Dateipfad',
        'file_too_large': 'Datei zu groß für die Verarbeitung',
        'batch_processing_complete': 'Stapelverarbeitung abgeschlossen',
        'health_check_passed': 'Systemgesundheitsprüfung bestanden',
        'cache_performance': 'Cache-Leistungsstatistiken'
    },
    'ja': {
        'processing_document': 'ドキュメントを処理中',
        'document_processed': 'ドキュメントの処理が正常に完了しました',
        'phi_detected': 'PHIエンティティが検出されました',
        'compliance_score': 'コンプライアンススコア',
        'error_processing': 'ドキュメント処理エラー',
        'security_validation_failed': 'セキュリティ検証に失敗しました',
        'invalid_file_path': '無効なファイルパス',
        'file_too_large': 'ファイルが処理には大きすぎます',
        'batch_processing_complete': 'バッチ処理が完了しました',
        'health_check_passed': 'システムヘルスチェックに合格しました',
        'cache_performance': 'キャッシュパフォーマンス統計'
    },
    'zh': {
        'processing_document': '正在处理文档',
        'document_processed': '文档处理成功',
        'phi_detected': '检测到PHI实体',
        'compliance_score': '合规评分',
        'error_processing': '文档处理错误',
        'security_validation_failed': '安全验证失败',
        'invalid_file_path': '无效的文件路径',
        'file_too_large': '文件太大无法处理',
        'batch_processing_complete': '批处理完成',
        'health_check_passed': '系统健康检查通过',
        'cache_performance': '缓存性能统计'
    }
}


class I18nManager:
    """Internationalization manager for multi-language support."""
    
    def __init__(self, default_language: str = 'en'):
        """Initialize i18n manager."""
        self.current_language = default_language
        self.translations = DEFAULT_TRANSLATIONS.copy()
        self._load_external_translations()
    
    def _load_external_translations(self) -> None:
        """Load translations from external files if available."""
        translations_dir = Path(__file__).parent / 'translations'
        if not translations_dir.exists():
            return
            
        for lang_code in SUPPORTED_LANGUAGES.keys():
            lang_file = translations_dir / f'{lang_code}.json'
            if lang_file.exists():
                try:
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        external_translations = json.load(f)
                        if lang_code not in self.translations:
                            self.translations[lang_code] = {}
                        self.translations[lang_code].update(external_translations)
                except Exception:
                    # Ignore errors loading external translations
                    pass
    
    def set_language(self, language_code: str) -> bool:
        """Set the current language."""
        if language_code in SUPPORTED_LANGUAGES:
            self.current_language = language_code
            return True
        return False
    
    def get_language(self) -> str:
        """Get the current language code."""
        return self.current_language
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported languages."""
        return SUPPORTED_LANGUAGES.copy()
    
    def t(self, key: str, **kwargs) -> str:
        """Translate a message key to the current language.
        
        Args:
            key: Translation key
            **kwargs: Format parameters for string interpolation
            
        Returns:
            Translated string with parameters interpolated
        """
        # Get translation for current language
        lang_dict = self.translations.get(self.current_language, {})
        
        # Fallback to English if key not found in current language
        if key not in lang_dict:
            lang_dict = self.translations.get('en', {})
        
        # Use key as fallback if no translation found
        message = lang_dict.get(key, key)
        
        # Format string with parameters if provided
        if kwargs:
            try:
                message = message.format(**kwargs)
            except (KeyError, ValueError):
                # Return unformatted message if formatting fails
                pass
        
        return message
    
    def detect_system_language(self) -> str:
        """Detect system language from environment variables."""
        # Check common environment variables
        for env_var in ['LANG', 'LANGUAGE', 'LC_ALL', 'LC_MESSAGES']:
            lang_env = os.environ.get(env_var, '')
            if lang_env:
                # Extract language code (first 2 characters)
                lang_code = lang_env[:2].lower()
                if lang_code in SUPPORTED_LANGUAGES:
                    return lang_code
        
        # Default to English
        return 'en'


# Global i18n manager instance
_global_i18n = None

def get_i18n_manager() -> I18nManager:
    """Get the global i18n manager instance."""
    global _global_i18n
    if _global_i18n is None:
        _global_i18n = I18nManager()
        # Auto-detect system language
        system_lang = _global_i18n.detect_system_language()
        _global_i18n.set_language(system_lang)
    return _global_i18n

def translate(key: str, **kwargs) -> str:
    """Global translation function."""
    return get_i18n_manager().t(key, **kwargs)

def set_global_language(language_code: str) -> bool:
    """Set the global language."""
    return get_i18n_manager().set_language(language_code)

def get_current_language() -> str:
    """Get the current global language."""
    return get_i18n_manager().get_language()