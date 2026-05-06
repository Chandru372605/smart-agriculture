"""
AgroSense — Prediction History API
GET  /api/history          → paginated list (filter by module, search)
GET  /api/history/stats    → aggregated stats per module
DELETE /api/history        → clear all records
DELETE /api/history/<id>   → delete one record
"""
from flask import Blueprint, request, jsonify
from backend.models.db_models import SessionLocal, PredictionLog
from sqlalchemy import func

history_bp = Blueprint('history', __name__)


@history_bp.route('/history', methods=['GET'])
def get_history():
    db     = SessionLocal()
    try:
        page    = int(request.args.get('page',   1))
        limit   = min(int(request.args.get('limit', 50)), 200)
        module  = request.args.get('module', '').strip()
        search  = request.args.get('search', '').strip()
        sort    = request.args.get('sort', 'newest')   # newest | oldest

        q = db.query(PredictionLog)
        if module:
            q = q.filter(PredictionLog.module == module)
        if search:
            q = q.filter(
                PredictionLog.summary.ilike(f'%{search}%') |
                PredictionLog.result.ilike(f'%{search}%')
            )
        q = q.order_by(
            PredictionLog.created_at.desc() if sort == 'newest'
            else PredictionLog.created_at.asc()
        )
        total   = q.count()
        records = q.offset((page - 1) * limit).limit(limit).all()
        return jsonify({
            'total':   total,
            'page':    page,
            'limit':   limit,
            'records': [r.to_dict() for r in records],
        })
    finally:
        db.close()


@history_bp.route('/history/stats', methods=['GET'])
def get_stats():
    db = SessionLocal()
    try:
        rows = (
            db.query(PredictionLog.module, func.count(PredictionLog.id).label('count'))
            .group_by(PredictionLog.module)
            .all()
        )
        total = db.query(func.count(PredictionLog.id)).scalar()
        return jsonify({
            'total': total,
            'by_module': [{'module': r.module, 'count': r.count} for r in rows],
        })
    finally:
        db.close()


@history_bp.route('/history', methods=['DELETE'])
def clear_history():
    db = SessionLocal()
    try:
        deleted = db.query(PredictionLog).delete()
        db.commit()
        return jsonify({'deleted': deleted})
    finally:
        db.close()


@history_bp.route('/history/<int:record_id>', methods=['DELETE'])
def delete_one(record_id):
    db = SessionLocal()
    try:
        rec = db.query(PredictionLog).filter(PredictionLog.id == record_id).first()
        if not rec:
            return jsonify({'error': 'Record not found'}), 404
        db.delete(rec)
        db.commit()
        return jsonify({'deleted': record_id})
    finally:
        db.close()
